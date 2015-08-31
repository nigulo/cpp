#define BOOST_NO_CXX11_SCOPED_ENUMS

#include "D2.h"
#include "BinaryDataLoader.h"
#include "TextDataLoader.h"
#include "utils.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <math.h>
#include <sstream>
#include <ctime>
#include <memory>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "mpi.h"
#ifdef _OPENACC
#include <openacc.h>
#else
#include <omp.h>
#endif

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace utils;

int procId;
int numProc;
time_t currentTime;

#define DIFF_NORMS_FILE_PREFIX "diffnorms"
#define DIFF_NORMS_FILE_SUFFIX ".csv"
#define DIFF_NORMS_FILE (DIFF_NORMS_FILE_PREFIX DIFF_NORMS_FILE_SUFFIX)
#define PARAMETERS_FILE_PREFIX "parameters"
#define PARAMETERS_FILE_SUFFIX ".txt"
#define PARAMETERS_FILE (PARAMETERS_FILE_PREFIX PARAMETERS_FILE_SUFFIX)

template<typename T> string vecToStr(const vector<T>& vec) {
	stringstream ss;
	bool first = true;
	for (auto t : vec) {
		if (!first) {
			ss << ",";
		}
		ss << t;
		first = false;
	}
	return ss.str();
}

template<> string vecToStr<pair<unsigned, unsigned>>(const vector<pair<unsigned, unsigned>>& vec) {
	stringstream ss;
	bool first = true;
	for (auto t : vec) {
		if (!first) {
			ss << ",";
		}
		ss << "{" << get<0>(t) << "," << get<1>(t) << "}";
		first = false;
	}
	return ss.str();
}


template<typename T> string vecVecToStr(const vector<vector<T>>& vec) {
	stringstream ss;
	bool first = true;
	for (auto& v : vec) {
		if (!first) {
			ss << ",";
		}
		ss << "{" << vecToStr(v) << "}";
		first = false;
	}
	return ss.str();
}


int main(int argc, char *argv[]) {

	MPI::Init (argc, argv);
	numProc = MPI::COMM_WORLD.Get_size();
	procId = MPI::COMM_WORLD.Get_rank();

#ifdef _OPENACC
	// It's recommended to perform one-off initialization of GPU-devices before any OpenACC regions
	acc_init(acc_get_device_type()) ; // from openacc.h
#endif

	if (procId == 0) {
		if (argc == 2 && string("-h") == argv[1]) {
			cout << "Usage: ./D2 [paramfile]\nparamfile defaults to " << PARAMETERS_FILE << endl;
			return EXIT_FAILURE;
		}
	}
	currentTime = time(nullptr);
	string paramFileName = argc > 1 ? argv[1] : PARAMETERS_FILE;

	if (procId == 0) {
		if (!exists(PARAMETERS_FILE)) {
			cout << "Cannot find " << PARAMETERS_FILE << endl;
			return EXIT_FAILURE;
		}
	}
	map<string, string> params = Utils::ReadProperties(paramFileName);

	string filePath = Utils::FindProperty(params, string("filePath") + to_string(procId) , "");
	assert(filePath.size() > 0);
	assert(exists(filePath));
	cout << "Rank: " << procId << " file: " << filePath << endl;
	bool binary = Utils::FindIntProperty(params, "binary", 0);
	unsigned bufferSize = Utils::FindIntProperty(params, "bufferSize", 0);

	string strDims = Utils::FindProperty(params, "dims", "1");
	vector<string> dimsStr;
	vector<unsigned> dims;
	boost::split(dimsStr, strDims, boost::is_any_of(","), boost::token_compress_on);
	for (vector<string>::iterator it = dimsStr.begin() ; it != dimsStr.end(); ++it) {
		if ((*it).length() != 0) {
			dims.push_back(stoi(*it));
		}
	}

	string strNumProcs = Utils::FindProperty(params, "numProcs", "1");
	vector<string> numProcsStr;
	vector<unsigned> numProcs;
	boost::split(numProcsStr, strNumProcs, boost::is_any_of(","), boost::token_compress_on);
	for (vector<string>::iterator it = numProcsStr.begin() ; it != numProcsStr.end(); ++it) {
		if ((*it).length() != 0) {
			numProcs.push_back(stoi(*it));
		}
	}

	assert(numProcs.size() == dims.size());
	assert(numProc == accumulate(numProcs.begin(), numProcs.end(), 1, multiplies<unsigned>()));

	vector<unsigned> dimsPerProc;
	for (unsigned i = 0; i < dims.size(); i++) {
		assert(dims[i] % numProcs[i] == 0);
		dimsPerProc.push_back(dims[i] / numProcs[i]);
	}


	vector<vector<pair<unsigned, unsigned>>> regions;
	string strRegions = Utils::FindProperty(params, "regions", "");
	if (!strRegions.empty()) {
		for (string strRegion : Utils::SplitByChars(strRegions, ";")) {
			vector<pair<unsigned, unsigned>> region;
			int i = 0;
			for (string strMinMax : Utils::SplitByChars(strRegion, ",", false)) {
				vector<string> minMaxStrs = Utils::SplitByChars(strMinMax, "-", false);
				assert(minMaxStrs.size() == 2);
				unsigned min = stoi(minMaxStrs[0]);
				unsigned procMin = procId * dimsPerProc[i];
				unsigned max = stoi(minMaxStrs[1]);
				unsigned procMax = (procId + 1) * dimsPerProc[i] - 1;
				if (min < procMin) {
					min = procMin;
				} else if (min > procMax) {
					min = procMax;
				} else {
					min %= dimsPerProc[i];
				}
				if (max > procMax) {
					max = procMax;
				} else if (max < procMin) {
					max = procMin;
				} else {
					max %= dimsPerProc[i];
				}
				region.push_back({min, max});
				i++;
			}
			assert(region.size() == dims.size());
			regions.push_back(region);
		}
	}
	/*
	string strMins = Utils::FindProperty(params, "mins", "0");
	vector<string> minsStr;
	vector<unsigned> mins;
	boost::split(minsStr, strMins, boost::is_any_of(","), boost::token_compress_on);
	for (vector<string>::iterator it = minsStr.begin() ; it != minsStr.end(); ++it) {
		if ((*it).length() != 0) {
			mins.push_back(stoi(*it));
		}
	}
	assert(mins.size() <= dims.size());
	for (unsigned i = 0; i < mins.size(); i++) {
		unsigned procMin = procId * dimsPerProc[i];
		if (mins[i] < procMin) {
			mins[i] = procMin;
		} else {
			mins[i] %= dimsPerProc[i];
		}
	}

	string strMaxs = Utils::FindProperty(params, "maxs", to_string(dims[0] - 1));
	vector<string> maxsStr;
	vector<unsigned> maxs;
	boost::split(maxsStr, strMaxs, boost::is_any_of(","), boost::token_compress_on);
	for (vector<string>::iterator it = maxsStr.begin() ; it != maxsStr.end(); ++it) {
		if ((*it).length() != 0) {
			maxs.push_back(stoi(*it));
		}
	}
	assert(maxs.size() <= dims.size());
	for (unsigned i = 0; i < maxs.size(); i++) {
		unsigned procMax = (procId + 1) * dimsPerProc[i];
		if (maxs[i] >= procMax) {
			maxs[i] = procMax;
		} else {
			maxs[i] %= dimsPerProc[i];
		}
	}
	*/

	unsigned totalNumVars = Utils::FindIntProperty(params, "numVars", 1);

	string strVarIndices = Utils::FindProperty(params, "varIndices", "0");
	vector<string> varIndicesStr;
	vector<unsigned> varIndices;
	boost::split(varIndicesStr, strVarIndices, boost::is_any_of(","), boost::token_compress_on);
	for (vector<string>::iterator it = varIndicesStr.begin() ; it != varIndicesStr.end(); ++it) {
		if ((*it).length() != 0) {
			varIndices.push_back(stoi(*it));
		}
	}

	double minPeriod = Utils::FindDoubleProperty(params, "minPeriod", 2);
	double maxPeriod = Utils::FindDoubleProperty(params, "maxPeriod", 10);
	double minCoherence = Utils::FindDoubleProperty(params, "minCoherence", 3);
	double maxCoherence = Utils::FindDoubleProperty(params, "maxCoherence", 30);
	string modeStr = Utils::FindProperty(params, "mode", "GaussWithCosine");
	to_upper(modeStr);
	Mode mode;
	if (modeStr == "BOX") {

	} else if (modeStr == "GAUSS") {
	} else if (modeStr == "GAUSSWITHCOSINE") {
		mode = Mode::GaussWithCosine;
	} else {
		cerr << "Invalid mode" << endl;
		assert(false);
	}
	bool normalize = Utils::FindIntProperty(params, "normalize", 1);
	bool relative = Utils::FindIntProperty(params, "relative", 1);

	double tScale = Utils::FindDoubleProperty(params, "tScale", 1);

	string strVarScales = Utils::FindProperty(params, "varScales", "1");
	vector<string> varScalesStr;
	vector<double> varScales;
	boost::split(varScalesStr, strVarScales, boost::is_any_of(","), boost::token_compress_on);
	for (vector<string>::iterator it = varScalesStr.begin() ; it != varScalesStr.end(); ++it) {
		if ((*it).length() != 0) {
			varScales.push_back(stod(*it));
		}
	}
	assert(varScales.size() <= varIndices.size());
	if (varScales.size() < varIndices.size()) {
		if (procId == 0) {
			cout << "Replacing missing variable scales with 1" << endl;
		}
		while (varScales.size() < varIndices.size()) {
			varScales.push_back(1.0f);
		}
	}

	if (procId == 0) {
		cout << "----------------" << endl;
		cout << "Parameter values" << endl;
		cout << "----------------" << endl;
		cout << "numProc      " << numProc << endl;
		cout << "binary       " << binary << endl;
		cout << "bufferSize   " << bufferSize << endl;
		cout << "dims         " << vecToStr(dims) << endl;
		cout << "regions      " << vecVecToStr(regions) << endl;
		cout << "numVars      " << totalNumVars << endl;
		cout << "minPeriod    " << minPeriod << endl;
		cout << "maxPeriod    " << maxPeriod << endl;
		cout << "minCoherence " << minCoherence << endl;
		cout << "maxCoherence " << maxCoherence << endl;
		cout << "mode         " << mode << endl;
		cout << "normalize    " << normalize << endl;
		cout << "relative     " << relative << endl;
		cout << "tScale       " << tScale << endl;
		cout << "varIndices   " << vecToStr(varIndices) << endl;
		cout << "varScales    " << vecToStr(varScales) << endl;
		cout << "----------------" << endl;
	}
	DataLoader* dl;
	if (binary) {
		dl = new BinaryDataLoader(filePath, bufferSize, dimsPerProc, regions, totalNumVars, varIndices);
	} else {
		dl = new TextDataLoader(filePath, bufferSize, dimsPerProc, regions, totalNumVars, varIndices);
	}
	D2 d2(*dl, minPeriod, maxPeriod, minCoherence, maxCoherence, mode, normalize, relative, tScale, varScales);
	if (!exists(DIFF_NORMS_FILE)) {
		d2.CalcDiffNorms();
	} else {
		d2.LoadDiffNorms();
	}
	d2.Compute2DSpectrum();
	delete dl;
	if (procId == 0) {
		cout << "done!" << endl;
	}
	MPI::Finalize();
	return EXIT_SUCCESS;
}

#define square(x) ((x) * (x))

D2::D2(DataLoader& rDataLoader, double minPeriod, double maxPeriod,
		double minCoherence, double maxCoherence,
		Mode mode, bool normalize, bool relative,
		double tScale, const vector<double>& varScales) :
			mrDataLoader(rDataLoader),
			minCoherence(minCoherence),
			maxCoherence(maxCoherence),
			mode(mode),
			normalize(normalize),
			relative(relative),
			tScale(tScale),
			varScales(varScales),
			e1(rd()) {
	assert(varScales.size() == rDataLoader.GetVarIndices().size());

	double wmax = 1.0 / minPeriod;
	wmin = 1.0 / maxPeriod;

	dmin = minCoherence * (relative ? minPeriod : 1);
	dmax = maxCoherence * (relative ? maxPeriod : 1);
	dmaxUnscaled = dmax / tScale;

	if (dmax < dmin || minPeriod > maxPeriod) {
		throw "Check Arguments";
	}
	k = dmax > dmin ? coherenceGrid : 1;
	m = round(phaseBins * (dmax - dmin) * wmax);
	a = (m - 1.0) / (dmax - dmin);
	b = -dmin * a;
	delta = (dmax - dmin) / (m - 1);

	lp = round((wmax - wmin) * (dmax - dmin) / deltaPhi);
	step = (wmax - wmin) / (lp - 1);
	eps = epsilon;
	epslim = 1.0 - eps;
	ln2 = sqrt(log(2.0));
	lnp = ln2 / eps;


}

double D2::Criterion(double d, double w) {
	double tyv = 0;
	int tav = 0;
	switch (mode) {
	case Box:
		for (unsigned j = 0; j < td.size(); j++) {
			double dd = td[j];
			if (dd <= d) {
				double ph = dd * w - floor(dd * w);
				if (ph < 0.0) {
					ph = ph + 1;
				}
				if (ph < eps || ph > epslim) {
					tyv += ty[j];
					tav += ta[j];
				}
			}
		}
		break;
	case Gauss: //This is important, in td[] are precomputed sums of squares and counts.
	case GaussWithCosine:
		for (unsigned j = 0; j < td.size(); j++) {// to jj-1 do begin
			double dd = td[j];
			double ww;
			if (d > 0.0) {
				ww = exp(-square(ln2 * dd / d));
			} else {
				ww = 0.0;
			}
			double ph = dd * w - floor(dd * w);//Frac(dd*w);
			if (ph < 0.0) {
				ph = ph + 1;
			}
			if (ph > 0.5) {
				ph = 1.0 - ph;
			}
			bool closeInPhase = true;
			double wp;
			if (mode == Gauss) {
				closeInPhase = ph < eps || ph > epslim;
				wp = exp(-square(lnp * ph));
			} else {
				if (ph == 0.5) {
					wp = 0;
				} else if (ph == 0) {
					wp = 1;
				} else {
					wp = 0.5 * (cos(0.5 * M_PI / ph) + 1);
				}
				if (std::isnan(wp)) {
					wp = 0;
					cout << "wp is still nan" << endl;
				}
			}
			if (closeInPhase) {
				tyv += ww * wp * ty[j];
				tav += ww * wp * ta[j];
			}
		}
		break;
	}
	if (tav > 0) {
		return 0.5 * tyv / tav;
	} else {
		return 0.0;
	}
}

void MapTo01D(vector<double>& cum) {

	unique_ptr<double> min;
	unique_ptr<double> max;

	for (auto& val : cum) {
		if (!min.get() || val < *min.get()) {
			min.reset(new double(val));
		}
		if (!max.get() || val > *max.get()) {
			max.reset(new double(val));
		}
	}
	if (max.get() && min.get() && *max.get() > *min.get()) {
		double range = *max.get() - *min.get();
		for (auto& val: cum) {
			val = (val - *min.get()) / range;
		}

	} else {
		cout << "Cannot normalize" << endl;
	}

}

// Currently implemented as Frobenius norm
double D2::DiffNorm(const real y1[], const real y2[]) {
	double norm = 0;
#ifdef _OPENACC
	#pragma acc data copyin(y1[0:mrDataLoader.GetDim() * mrDataLoader.GetNumVars()], y2[0:mrDataLoader.GetDim() * mrDataLoader.GetNumVars()])
	#pragma acc parallel loop reduction(+:norm)
#else
	#pragma omp parallel for reduction(+:norm)
#endif
	for (unsigned i = 0; i < mrDataLoader.GetDim(); i++) {
		if (mrDataLoader.InRegion(i)) {
			auto offset = i * mrDataLoader.GetNumVars();
			for (unsigned j : mrDataLoader.GetVarIndices()) {
				auto index = offset + j;
				norm += square(y1[index] - y2[index]);
			}
		}
	}
	return norm;
}

#define TAG_TTY 1
#define TAG_TTA 2

bool D2::ProcessPage(DataLoader& dl1, DataLoader& dl2, vector<double>& tty, vector<int>& tta) {
	bool bootstrap = false;
	if (dl2.GetX(0) - dl1.GetX(dl1.GetPageSize() - 1) > dmaxUnscaled) {
		return false;
	}
	for (unsigned i = 0; i < dl1.GetPageSize(); i++) {
		unsigned j = 0;
		if (dl1.GetPage() == dl2.GetPage()) {
			j = i + 1;
		}
		int countNeeded = 0;
		int countTaken = 0;
		if (bootstrap) {
			for (; j < dl2.GetPageSize(); j++) {
				real d = (dl2.GetX(j) - dl1.GetX(i)) * tScale;
				if (d > dmax) {
					break;
				}
				countNeeded++;
			}

		} else {
			countNeeded  = dl2.GetPageSize() - j;
		}
		uniform_int_distribution<int> uniform_dist(0, countNeeded - 1);
		for (; j < dl2.GetPageSize(); j++) {
			real d = (dl2.GetX(j) - dl1.GetX(i)) * tScale;
			if (d > dmax || (bootstrap && countTaken >= countNeeded)) {
				break;
			}
			if (d >= dmin) {
				int kk = round(a * d + b);
				for (int counter = 0; counter < countNeeded; counter++) {
					bool take = true;
					if (bootstrap) {
						take = uniform_dist(e1) == counter;
					}
					if (take) {
						tty[kk] += DiffNorm(dl2.GetY(j), dl1.GetY(i));
						tta[kk]++;
						countTaken++;
					}
					if (!bootstrap) {
						break;
					}
				}
			}
		}
	}
	return true;
}


void D2::CalcDiffNorms() {
	if (procId == 0) {
		cout << "Calculating diffnorms..." << endl;
	}

	vector<double> tty(m, 0);
	vector<int> tta(m, 0);

	// Now comes precomputation of differences and counts. They are accumulated in two grids.
	if (procId == 0) {
		cout << "Loading data..." << endl;
	}
	while (mrDataLoader.Next()) {
		if (!ProcessPage(mrDataLoader, mrDataLoader, tty, tta)) {
			break;
		}
		DataLoader* dl2 = mrDataLoader.Clone();
		if (dl2) {
			do {
				if (!ProcessPage(mrDataLoader, *dl2, tty, tta)) {
					break;
				}
			} while (dl2->Next());
		}
		if (procId == 0) {
			cout << "Page " << mrDataLoader.GetPage() << " loaded." << endl;
		}
		delete dl2;
	}
	if (procId == 0) {
		cout << "Waiting for data from other processes..." << endl;
	}
	MPI::COMM_WORLD.Barrier();
	if (procId > 0) {
		cout << "Sending data from " << procId << "." << endl;
		//for (unsigned j = 0; j < m; j++) {
		//	cout << tty[j] << endl;
		//}
		MPI::COMM_WORLD.Send(tty.data(), tty.size(), MPI::DOUBLE, 0, TAG_TTY);
		MPI::COMM_WORLD.Send(tta.data(), tta.size(), MPI::INT, 0, TAG_TTA);
	} else {
		for (int i = 1; i < numProc; i++) {
			double ttyRecv[m];
			int ttaRecv[m];
			MPI::Status status;
			MPI::COMM_WORLD.Recv(ttyRecv, m,  MPI::DOUBLE, MPI_ANY_SOURCE, TAG_TTY, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received square differences from " << status.Get_source() << "." << endl;
			MPI::COMM_WORLD.Recv(ttaRecv, m,  MPI::INT, status.Get_source(), TAG_TTA, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received weights from " << status.Get_source() << "." << endl;
			for (unsigned j = 0; j < m; j++) {
				tty[j] += ttyRecv[j];
				assert(tta[j] == ttaRecv[j]);
				//tta[j] += ttaRecv[j];
				//cout << ttyRecv[j] << endl;
			}
		}
		// How many time differences was actually used?
		unsigned j = 0;
		for (unsigned i = 0; i < m; i++) {
			if (tta[i] > 0) {
				j++;
			}
		}
		cout << "j=" << j << endl;
		ta.assign(j, 0);
		ty.assign(j, 0);
		td.assign(j, 0);
		cout << "td.size()=" << td.size() << endl;
		double d = dmin;

		// Build final grids for periodicity search.

		j = 0;
		ofstream output(string(DIFF_NORMS_FILE_PREFIX) + "_" + to_string(currentTime) + DIFF_NORMS_FILE_SUFFIX);
		for (unsigned i = 0; i < m; i++) {
			if (tta[i] > 0) {
				td[j] = d;
				ty[j] = tty[i];
				ta[j] = tta[i];
				output << d << " " << ty[j] << " " << ta[j] << endl;
				j++;
			}
			d = d + delta;
		}
		output.close();
		copy_file(PARAMETERS_FILE, string(PARAMETERS_FILE_PREFIX) + "_" + to_string(currentTime) + PARAMETERS_FILE_SUFFIX);
	}

}

void D2::LoadDiffNorms() {
	if (procId == 0) {
		cout << "Loading diffnorms..." << endl;
		ifstream input(DIFF_NORMS_FILE);
		for (string line; getline(input, line);) {
			std::vector<std::string> words;
			boost::split(words, line, boost::is_any_of("\t "), boost::token_compress_on);
			for (vector<string>::iterator it = words.begin(); it != words.end();) {
				if ((*it).length() == 0) {
					it = words.erase(it);
				} else {
					it++;
				}
			}
			if (words.size() > 0 && words[0][0] == '#') {
				//cout << "Skipping comment line: " << line << endl;
			} else if (words.size() == 3) {
				try {
					td.push_back(stod(words[0]));
					ty.push_back(stod(words[1]));
					ta.push_back(stoi(words[2]));
				} catch (invalid_argument& ex) {
					cout << "Skipping line, invalid number: " << line << endl;
				}
			} else {
				cout << "Skipping line, invalid number of columns: " << line << endl;
			}
		}
		input.close();
	}
}

void D2::Compute2DSpectrum() {

	if (procId == 0) {
		cout << "lp = " << lp << endl;
		cout << "k = " << k << endl;
		cout << "m = " << m << endl;
		cout << "a = " << a << endl;
		cout << "b = " << b << endl;
		cout << "dmin = " << dmin << endl;
		cout << "dmax = " << dmax << endl;
		cout << "wmin = " << wmin << endl;
		cout << "delta = " << delta << endl;
		cout << "step = " << step << endl;

		ofstream output("phasedisp.csv");
		ofstream output_min("phasedisp_min.csv");
		ofstream output_max("phasedisp_max.csv");

		vector<double> cum(lp);
		cum.assign(lp, 0);
		// Basic cycle with printing for GnuPlot

		double deltac = maxCoherence > minCoherence ? (maxCoherence - minCoherence) / (k - 1) : 0;
		for (unsigned i = 0; i < k; i++) {
			double d = minCoherence + i * deltac;
			for (unsigned j = 0; j < lp; j++) {
				double w = wmin + j * step;
				double d1 = d;
				if (relative) {
					d1 = d / w;
				}
				double res=Criterion(d1, w);
				cum[j] = res;
			}

			// Spectrum in cum can be normalized

			if (normalize) {
				MapTo01D(cum);
			}

			vector<int> minima(0);
			unsigned dk = lp / 20;
			for (unsigned j = 0; j < lp; j++) {
				output << d << " " << (wmin + j * step) << " " << cum[j] << endl;
				if (d == minCoherence) {
					output_min << (wmin + j * step) << " " << cum[j] << endl;
				} else if (d == maxCoherence) {
					output_max << (wmin + j * step) << " " << cum[j] << endl;
				}

				if (j > dk - 1 && j < lp - dk - 1) {
					bool isMinimum = true;
					for (unsigned k1 = j - dk; k1 <= j + dk; k1++) {
						if (cum[j] > cum[k1]) {
							isMinimum = false;
							break;
						}
					}
					if (isMinimum) {
						vector<int>::iterator it = minima.begin();
						for (; it != minima.end(); ++it) {
							if (cum[j] < cum[(*it)]) {
								break;
							}
						}
						minima.insert(it, j);
					}
				}
			}
			for (unsigned k1 = 0; k1 < minima.size(); k1++) {
				if (k1 > 0) {
					//cout << " ";
				}
				//cout << wmin + minima[k1] * step;
			}
			//cout << endl;
		}
		output.close();
		output_min.close();
		output_max.close();
	}
}

