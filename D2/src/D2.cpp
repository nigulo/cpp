#define BOOST_NO_CXX11_SCOPED_ENUMS

#include "D2.h"
#include "BinaryDataLoader.h"
#include "TextDataLoader.h"
#include "utils/utils.h"

#include <utility>
#include <limits>
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
#ifndef _NOMPI
#include "mpi.h"
#endif
#ifdef _OPENACC
#include <openacc.h>
#else
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace utils;

int procId;
int numProc;
time_t currentTime;
bool saveDiffNorms;
bool saveParameters;

#define DIFF_NORMS_FILE_PREFIX "diffnorms"
#define DIFF_NORMS_FILE_SUFFIX ".csv"
#define DIFF_NORMS_FILE (DIFF_NORMS_FILE_PREFIX DIFF_NORMS_FILE_SUFFIX)
#define PARAMETERS_FILE_PREFIX "parameters"
#define PARAMETERS_FILE_SUFFIX ".txt"
#define PARAMETERS_FILE (PARAMETERS_FILE_PREFIX PARAMETERS_FILE_SUFFIX)

string paramFileName;

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

#ifdef _NOMPI
	numProc = 1;
	procId = 0;
#else
	MPI::Init(argc, argv);
	numProc = MPI::COMM_WORLD.Get_size();
	procId = MPI::COMM_WORLD.Get_rank();
#endif
#ifdef _OPENACC
	// It's recommended to perform one-off initialization of GPU-devices before any OpenACC regions
	acc_init(acc_get_device_type()) ; // from openacc.h
#endif

	if (procId == 0) {
		if (argc == 2 && string("-h") == argv[1]) {
			cout << "Usage: ./D2 [paramfile]\nparamfile defaults to " << PARAMETERS_FILE << endl;
			return EXIT_SUCCESS;
		}
	}
	currentTime = time(nullptr);
	paramFileName = argc > 1 ? argv[1] : PARAMETERS_FILE;

	if (procId == 0) {
		if (!exists(paramFileName)) {
			cout << "Cannot find " << paramFileName << endl;
			return EXIT_FAILURE;
		}
	}
	map<string, string> params = Utils::ReadProperties(paramFileName);
	double initMinPeriod = Utils::FindDoubleProperty(params, "minPeriod", 2);
	double initMaxPeriod = Utils::FindDoubleProperty(params, "maxPeriod", 10);
	double minCoherence = Utils::FindDoubleProperty(params, "minCoherence", 3);
	double maxCoherence = Utils::FindDoubleProperty(params, "maxCoherence", 30);
	string modeStr = Utils::FindProperty(params, "mode", "GaussWithCosine");
	to_upper(modeStr);
	Mode mode;
	if (modeStr == "BOX") {
		mode = Mode::Box;
	} else if (modeStr == "GAUSS") {
		mode = Mode::Gauss;
	} else if (modeStr == "GAUSSWITHCOSINE") {
		mode = Mode::GaussWithCosine;
	} else {
		cerr << "Invalid mode" << endl;
		assert(false);
	}
	bool normalize = Utils::FindIntProperty(params, "normalize", 0);
	bool relative = Utils::FindIntProperty(params, "relative", 1);
	bool removeSpurious = Utils::FindIntProperty(params, "removeSpurious", 0);

	double tScale = Utils::FindDoubleProperty(params, "tScale", 1);

	string strVarScales = Utils::FindProperty(params, "varScales", "1");
	vector<string> varScalesStr;
	vector<double> varScales;
	boost::split(varScalesStr, strVarScales, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = varScalesStr.begin() ; it != varScalesStr.end(); ++it) {
		if ((*it).length() != 0) {
			varScales.push_back(stod(*it));
		}
	}

	string strVarRanges = Utils::FindProperty(params, "varRanges", "");
	trim(strVarRanges);
	vector<pair<double, double>> varRanges;
	if (!strVarRanges.empty()) {
		vector<string> varRangeStrs = Utils::SplitByChars(strVarRanges, ",;", false);
		for (auto& varRange : varRangeStrs) {
			vector<string> minAndMax = Utils::SplitByChars(varRange, ":", false);
			assert(minAndMax.size() == 2);
			trim(minAndMax[0]);
			trim(minAndMax[1]);
			double min = minAndMax[0].empty() ? numeric_limits<double>::min() : stod(minAndMax[0]);
			double max = minAndMax[1].empty() ? numeric_limits<double>::max() : stod(minAndMax[1]);
			varRanges.push_back({min, max});
		}
	}

	int numIterations = Utils::FindIntProperty(params, "numIterations", 1);
	if (numIterations > 1 && numProc > 1) {
		cout << "Zooming not available in multiprocessor regime" << endl;
		return EXIT_FAILURE;
	}
	double zoomFactor = Utils::FindDoubleProperty(params, "zoomFactor", 10);

	assert(numIterations >= 1);
	assert(zoomFactor >= 1);

	if (procId == 0) {
		cout << "----------------" << endl;
		cout << "Parameter values" << endl;
		cout << "----------------" << endl;
		cout << "numProc        " << numProc << endl;
		cout << "minPeriod      " << initMinPeriod << endl;
		cout << "maxPeriod      " << initMaxPeriod << endl;
		cout << "minCoherence   " << minCoherence << endl;
		cout << "maxCoherence   " << maxCoherence << endl;
		cout << "mode           " << mode << endl;
		cout << "normalize      " << normalize << endl;
		cout << "relative       " << relative << endl;
		cout << "removeSpurious " << removeSpurious << endl;
		cout << "tScale         " << tScale << endl;
		cout << "varScales      " << vecToStr(varScales) << endl;
		cout << "numIterations  " << numIterations << endl;
		cout << "zoomFactor     " << zoomFactor << endl;
		cout << "----------------" << endl;
	}
	string filePathsStr = Utils::FindProperty(params, string("filePath") + to_string(procId), "");
	vector<string> filePaths;
	boost::split(filePaths, filePathsStr, boost::is_any_of(",;"), boost::token_compress_on);

	string outputFilePrefixesStr = Utils::FindProperty(params, string("outputFilePath"), "");
	vector<string> outputFilePrefixes;
	if (outputFilePrefixesStr.length() > 0) {
		boost::split(outputFilePrefixes, outputFilePrefixesStr, boost::is_any_of(",;"), boost::token_compress_on);
	}
	saveDiffNorms = Utils::FindIntProperty(params, "saveDiffNorms", 1);
	saveParameters = Utils::FindIntProperty(params, "saveParameters", 1);

	bool binary = Utils::FindIntProperty(params, "binary", 0);
	unsigned bufferSize = Utils::FindIntProperty(params, "bufferSize", 0);

	string strDims = Utils::FindProperty(params, "dims", "1");
	vector<string> dimsStr;
	vector<unsigned> dims;
	boost::split(dimsStr, strDims, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = dimsStr.begin() ; it != dimsStr.end(); ++it) {
		if ((*it).length() != 0) {
			dims.push_back(stoi(*it));
		}
	}

	string strNumProcs = Utils::FindProperty(params, "numProcs", "1");
	vector<string> numProcsStr;
	vector<unsigned> numProcs;
	boost::split(numProcsStr, strNumProcs, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = numProcsStr.begin() ; it != numProcsStr.end(); ++it) {
		if ((*it).length() != 0) {
			numProcs.push_back(stoi(*it));
		}
	}

	assert(numProcs.size() == dims.size());
	assert(numProc == accumulate(numProcs.begin(), numProcs.end(), 1, multiplies<unsigned>()));

	vector<unsigned> dimsPerProc;
	vector<unsigned> procIds;
	for (unsigned i = 0; i < dims.size(); i++) {
		assert(dims[i] % numProcs[i] == 0);
		dimsPerProc.push_back(dims[i] / numProcs[i]);
		procIds.push_back(procId % numProcs[i]);
	}


	vector<vector<pair<unsigned, unsigned>>> regions;
	string strRegions = Utils::FindProperty(params, "regions", "");
	if (!strRegions.empty()) {
		for (string strRegion : Utils::SplitByChars(strRegions, ";")) {
			vector<pair<unsigned, unsigned>> region;
			int i = 0;
			for (string strMinMax : Utils::SplitByChars(strRegion, ",", false)) {
				vector<string> minMaxStrs = Utils::SplitByChars(strMinMax, ":-", false);
				assert(minMaxStrs.size() == 2);
				unsigned min = stoi(minMaxStrs[0]);
				unsigned procMin = procIds[i] * dimsPerProc[i];
				unsigned max = stoi(minMaxStrs[1]);
				unsigned procMax = (procIds[i] + 1) * dimsPerProc[i] - 1;
				//cout << procId << " min, max, procMin, procMax: " << min << ", " << max << ", " << procMin << ", " << procMax << endl;
				if (min <= procMax && max >= procMin) {
					if (min < procMin) {
						min = 0;
					} else {
						min %= dimsPerProc[i];
					}
					if (max > procMax) {
						max = dimsPerProc[i] - 1;
					} else {
						max %= dimsPerProc[i];
					}
					region.push_back({min, max});
				} else {
					region.clear();
					break;
				}
				i++;
			}
			if (region.size() == dims.size()) {
				//cout << procId << " region" << endl;
				regions.push_back(region);
			}
		}
	}

	unsigned totalNumVars = Utils::FindIntProperty(params, "numVars", 1);

	string strVarIndices = Utils::FindProperty(params, "varIndices", "0");
	vector<string> varIndicesStr;
	vector<unsigned> varIndices;
	boost::split(varIndicesStr, strVarIndices, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = varIndicesStr.begin() ; it != varIndicesStr.end(); ++it) {
		if ((*it).length() != 0) {
			varIndices.push_back(stoi(*it));
		}
	}
	if (procId == 0) {
		cout << "binary         " << binary << endl;
		cout << "bufferSize     " << bufferSize << endl;
		cout << "dims           " << vecToStr(dims) << endl;
		cout << "regions        " << vecVecToStr(regions) << endl;
		cout << "numVars        " << totalNumVars << endl;
		cout << "varIndices     " << vecToStr(varIndices) << endl;
		cout << "----------------" << endl;
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
	if (varRanges.size() < varIndices.size()) {
		if (procId == 0) {
			cout << "Replacing missing variable ranges with min and max" << endl;
		}
		while (varRanges.size() < varIndices.size()) {
			varRanges.push_back({numeric_limits<double>::min(), numeric_limits<double>::max()});
		}
	}

	int filePathIndex = 0;
	for (auto& filePath : filePaths) {
		assert(filePath.size() > 0);
		double minPeriod = initMinPeriod;
		double maxPeriod = initMaxPeriod;
		for (int i = 0; i < numIterations; i++) {
			DataLoader* dl = nullptr;
			if (exists(filePath)) {
				cout << "Rank: " << procId << " file: " << filePath << endl;

				if (binary) {
					dl = new BinaryDataLoader(filePath, bufferSize, dimsPerProc, regions, totalNumVars, varIndices);
				} else {
					dl = new TextDataLoader(filePath, bufferSize, dimsPerProc, regions, totalNumVars, varIndices);
				}

			}

			D2 d2(dl, minPeriod, maxPeriod, minCoherence, maxCoherence, mode, normalize, relative, tScale, varScales, varRanges, removeSpurious);
			if (!exists(DIFF_NORMS_FILE)) {
				assert(dl); // dataLoader must be present in case diffnorms are not calculated yet
				d2.CalcDiffNorms(filePathIndex);
			} else {
				d2.LoadDiffNorms(filePathIndex);
			}
			string outputFilePrefix = "phasedisp";
			if (filePathIndex > 0) {
				outputFilePrefix += to_string(filePathIndex);
			}
			if ((int) outputFilePrefixes.size() > filePathIndex) {
				outputFilePrefix = outputFilePrefixes[filePathIndex];
			}
			double d2Freq = d2.Compute2DSpectrum(outputFilePrefix);
			if (dl) {
				delete dl;
			}
			if (i < numIterations - 1) {
				if (d2Freq == 0) {
					cout << "Finish zooming, no minima found"  << endl;
					break;
				}
				double d2Period = 1 / d2Freq;
				double periodRange = (maxPeriod - minPeriod) / 2 / zoomFactor;
				minPeriod = d2Period - periodRange;
				maxPeriod = d2Period + periodRange;
				if (minPeriod < initMinPeriod) {
					minPeriod = initMinPeriod;
				}
				if (maxPeriod > initMaxPeriod) {
					maxPeriod = initMaxPeriod;
				}
				cout << "d2 minimum period=" << d2Period << endl;
				cout << "new minPeriod=" << minPeriod << endl;
				cout << "new maxPeriod=" << maxPeriod << endl;
			}
		}
		filePathIndex++;
#ifndef _NOMPI
		MPI::COMM_WORLD.Barrier();
#endif
	}
	if (procId == 0) {
		cout << "done!" << endl;
	}
#ifndef _NOMPI
	MPI::Finalize();
#endif
	return EXIT_SUCCESS;
}

#define square(x) ((x) * (x))

D2::D2(DataLoader* pDataLoader, double minPeriod, double maxPeriod,
		double minCoherence, double maxCoherence,
		Mode mode, bool normalize, bool relative,
		double tScale, const vector<double>& varScales, const vector<pair<double, double>>& varRanges, bool removeSpurious) :
			mpDataLoader(pDataLoader),
			minCoherence(minCoherence),
			maxCoherence(maxCoherence),
			mode(mode),
			normalize(normalize),
			relative(relative),
			tScale(tScale),
			varScales(varScales),
			varRanges(varRanges),
			removeSpurious(removeSpurious),
			e1(rd()) {
	if (pDataLoader) {
		assert(varScales.size() == pDataLoader->GetVarIndices().size());
	}

	double wmax = 1.0 / minPeriod;
	wmin = 1.0 / maxPeriod;

	dmin = minCoherence * (relative ? minPeriod : 1);
	dmax = maxCoherence * (relative ? maxPeriod : 1);
	dbase = dmin / 10;
	dmaxUnscaled = dmax / tScale;

	if (dmax < dmin || minPeriod > maxPeriod) {
		throw "Check Arguments";
	}
	numCoherences = dmax > dmin ? coherenceGrid : 1; // output precision in coherence
	numCoherenceBins = round(phaseBins * (dmax - dbase) * wmax);
	coherenceBinSize = (dmax - dbase) / (numCoherenceBins - 1);
	a = (numCoherenceBins - 1.0) / (dmax - dbase);
	b = -dbase * a;

	freqStep = (wmax - wmin) / (numFreqs - 1);
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
				if (ph < 0) {
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
				if (ph >= 0.25) {
					wp = 0;
				} else if (ph == 0) {
					wp = 1;
				} else {
					wp = 0.5 * (cos(2 * M_PI * ph) + 1);
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
	if (tav > 0 && varSum > 0) {
		return 0.5 * tyv / tav / varSum;
	} else {
		cout << "tav=" << tav << endl;
		return 0.0;
	}
}

void Normalize(vector<pair<double, double>>& spec) {

	unique_ptr<double> min;
	unique_ptr<double> max;

	for (auto& val : spec) {
		if (!min || val.second < *min) {
			min.reset(new double(val.second));
		}
		if (!max || val.second > *max) {
			max.reset(new double(val.second));
		}
	}
	if (max && min && *max > *min) {
		double range = *max - *min;
		for (auto& val : spec) {
			val = {val.first, (val.second - *min) / range};
		}

	} else {
		cout << "Cannot normalize" << endl;
	}

}

// Currently implemented as Frobenius norm
double D2::DiffNorm(const real y1[], const real y2[]) {
	double norm = 0;
#ifdef _OPENACC
	#pragma acc data copyin(y1[0:mpDataLoader->GetDim() * mpDataLoader->GetNumVars()], y2[0:mpDataLoader->GetDim() * mpDataLoader->GetNumVars()])
	#pragma acc parallel loop reduction(+:norm)
#else
#ifdef _OPENMP
	#pragma omp parallel for reduction(+:norm)
#endif
#endif
	for (unsigned j = 0; j < mpDataLoader->GetNumVars(); j++) {
		auto offset = j * mpDataLoader->GetDim();
		auto varScale = varScales[j];
		auto varRange = varRanges[j];
		if (varScale != 1.0f) {
			for (unsigned i = 0; i < mpDataLoader->GetDim(); i++) {
				if (mpDataLoader->IsInRegion(i)) {
					auto index = offset + i;
					if (y1[index] >= varRange.first && y1[index] <= varRange.second
							&& y2[index] >= varRange.first && y2[index] <= varRange.second) {
						norm += square((y1[index] - y2[index]) * varScale);
					}
				}
			}
		} else {
			for (unsigned i = 0; i < mpDataLoader->GetDim(); i++) {
				if (mpDataLoader->IsInRegion(i)) {
					auto index = offset + i;
					if (y1[index] >= varRange.first && y1[index] <= varRange.second
							&& y2[index] >= varRange.first && y2[index] <= varRange.second) {
						norm += square(y1[index] - y2[index]);
					}
				}
			}
		}
	}
	return norm;
}



#define TAG_TTY 1
#define TAG_TTA 2
#define TAG_VAR 3

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
		//if (procId == 0) {
		//	cout << "Time :" << dl1.GetX(i) << endl;
		//}
		for (; j < dl2.GetPageSize(); j++) {
			if (dl2.GetX(j) * tScale > maxX) {
				maxX = dl2.GetX(j) * tScale;
			}
			real d = (dl2.GetX(j) - dl1.GetX(i)) * tScale;
			if (d > dmax || (bootstrap && countTaken >= countNeeded)) {
				break;
			}
			if (d >= dbase) {
				int kk = round(a * d + b);
				for (int counter = 0; counter < countNeeded; counter++) {
					bool take = true;
					if (bootstrap) {
						take = uniform_dist(e1) == counter;
					}
					if (take) {
						auto dy2 = DiffNorm(dl2.GetY(j), dl1.GetY(i));
						tty[kk] += dy2;
						tta[kk]++;
						//cout << "tta[" << kk << "]=" << tta[kk] << endl;
						//cout << "tty[" << kk << "]=" << tty[kk] << endl;
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


void D2::CalcDiffNorms(int filePathIndex) {
	assert(mpDataLoader);
	if (procId == 0) {
		cout << "Calculating diffnorms..." << endl;
	}

	vector<double> tty(numCoherenceBins, 0);
	vector<int> tta(numCoherenceBins, 0);

	// Now comes precomputation of differences and counts. They are accumulated in two grids.
	if (procId == 0) {
		cout << "Loading data..." << endl;
	}
	unsigned size = mpDataLoader->GetNumVars() * mpDataLoader->GetDim();
	double ySum[size];
	double y2Sum[size];
	int n = 0;
	for (unsigned i = 0; i < size; i++) {
		ySum[i] = 0;
		y2Sum[i] = 0;
	}
	while (mpDataLoader->Next()) {
		for (unsigned i = 0; i < mpDataLoader->GetPageSize(); i++) {
			n++;
			auto y = mpDataLoader->GetY(i);
			// ------------------------------------------
			// This calculation must be redesigned
			for (unsigned j = 0; j < mpDataLoader->GetNumVars(); j++) {
				auto offset = j * mpDataLoader->GetDim();
				auto varScale = varScales[j];
				auto varRange = varRanges[j];
				if (varScale != 1.0f) {
					for (unsigned i = 0; i < mpDataLoader->GetDim(); i++) {
						if (mpDataLoader->IsInRegion(i)) {
							auto index = offset + i;
							auto yScaled = y[index] * varScale;
							if (yScaled >= varRange.first && yScaled <= varRange.second) {
								ySum[index] += yScaled;
								y2Sum[index] += yScaled * yScaled;
							}
						}
					}
				} else {
					for (unsigned i = 0; i < mpDataLoader->GetDim(); i++) {
						if (mpDataLoader->IsInRegion(i)) {
							auto index = offset + i;
							if (y[index] >= varRange.first && y[index] <= varRange.second) {
								ySum[index] += y[index];
								y2Sum[index] += y[index] * y[index];
							}
						}
					}
				}
			}
			// ------------------------------------------
		}
		if (!ProcessPage(*mpDataLoader, *mpDataLoader, tty, tta)) {
			break;
		}
		DataLoader* dl2 = mpDataLoader->Clone();
		if (dl2) {
			do {
				if (!ProcessPage(*mpDataLoader, *dl2, tty, tta)) {
					break;
				}
			} while (dl2->Next());
		}
		if (procId == 0) {
			cout << "Page " << mpDataLoader->GetPage() << " loaded." << endl;
		}
		delete dl2;
	}
	varSum = 0;
	for (unsigned i = 0; i < size; i++) {
		varSum += (y2Sum[i] - (ySum[i] * ySum[i]) / n) / (n - 1);
	}
	if (procId == 0) {
		cout << "Waiting for data from other processes..." << endl;
	}
#ifndef _NOMPI
	MPI::COMM_WORLD.Barrier();
#endif
	if (procId > 0) {
		cout << "Sending data from " << procId << "." << endl;
		//for (unsigned j = 0; j < m; j++) {
		//	cout << tty[j] << endl;
		//}
#ifndef _NOMPI
		MPI::COMM_WORLD.Send(tty.data(), tty.size(), MPI::DOUBLE, 0, TAG_TTY);
		MPI::COMM_WORLD.Send(tta.data(), tta.size(), MPI::INT, 0, TAG_TTA);
		MPI::COMM_WORLD.Send(&varSum, 1, MPI::DOUBLE, 0, TAG_VAR);
#endif
	} else {
#ifndef _NOMPI
		for (int i = 1; i < numProc; i++) {
			double ttyRecv[numCoherenceBins];
			int ttaRecv[numCoherenceBins];
			MPI::Status status;
			MPI::COMM_WORLD.Recv(ttyRecv, numCoherenceBins,  MPI::DOUBLE, MPI_ANY_SOURCE, TAG_TTY, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received square differences from " << status.Get_source() << "." << endl;
			MPI::COMM_WORLD.Recv(ttaRecv, numCoherenceBins,  MPI::INT, status.Get_source(), TAG_TTA, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received weights from " << status.Get_source() << "." << endl;
			for (unsigned j = 0; j < numCoherenceBins; j++) {
				tty[j] += ttyRecv[j];
				assert(tta[j] == ttaRecv[j]);
				//tta[j] += ttaRecv[j];
				//cout << ttyRecv[j] << endl;
			}

			double varSumRecv;
			MPI::COMM_WORLD.Recv(&varSumRecv, 1,  MPI::DOUBLE, status.Get_source(), TAG_VAR, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received variance sum " << status.Get_source() << "." << endl;
			for (unsigned j = 0; j < numCoherenceBins; j++) {
				varSum += varSumRecv;
			}
		}
#endif
		cout << "varSum: " << varSum << endl;
		// How many time differences was actually used?
		unsigned j = 0;
		for (unsigned i = 0; i < numCoherenceBins; i++) {
			if (tta[i] > 0) {
				j++;
			}
		}
		cout << "j=" << j << endl;
		ta.assign(j, 0);
		ty.assign(j, 0);
		td.assign(j, 0);
		cout << "td.size()=" << td.size() << endl;

		// Build final grids for periodicity search.

		j = 0;
		for (unsigned i = 0; i < numCoherenceBins; i++) {
			double d = dbase + i * coherenceBinSize;
			if (tta[i] > 0) {
				td[j] = d;
				ty[j] = tty[i];
				ta[j] = tta[i];
				j++;
			}
		}
		if (saveDiffNorms) {
			string diffNormsFilePrefix = string(DIFF_NORMS_FILE_PREFIX);
			if (filePathIndex > 0) {
				diffNormsFilePrefix += to_string(filePathIndex);
			}
			ofstream output(diffNormsFilePrefix + "_" + to_string(currentTime) + DIFF_NORMS_FILE_SUFFIX);
			output << varSum << endl;
			for (unsigned i = 0; i < j; i++) {
				output << td[i] << " " << ty[i] << " " << ta[i] << endl;
			}
			output.close();
		}
		if (saveParameters && filePathIndex == 0) {
			copy_file(paramFileName, string(PARAMETERS_FILE_PREFIX) + "_" + to_string(currentTime) + PARAMETERS_FILE_SUFFIX);
		}
	}

}

void D2::LoadDiffNorms(int filePathIndex) {
	if (procId == 0) {
		varSum = 1; // Assuming unit variance by default
		cout << "Loading diffnorms..." << endl;
		string diffNormsFile = DIFF_NORMS_FILE;
		if (filePathIndex > 0) {
			diffNormsFile = string(DIFF_NORMS_FILE_PREFIX) + to_string(filePathIndex)  + DIFF_NORMS_FILE_SUFFIX;
		}
		ifstream input(diffNormsFile);
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
			} else if (words.size() == 1) {
				// loading varSum
				varSum = stod(words[0]);
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

vector<pair<double, double>> getLocalMinima(const vector<pair<double, double>>& spec) {
	vector<pair<double, double>> retVal;
	int start = 0;
	for (int i = 1; i < (int) spec.size() - 1; i++) {
		if (spec[i].second < spec[i - 1].second) {
			start = i;
		}
		if (start > 0 && spec[i].second < spec[i + 1].second) {
			retVal.push_back(spec[(i + start) / 2]);
			start = 0;
		}
	}
	return retVal;
}

vector<double> getSpurious(double p0) {
	double p = 1.0;
	double k1[] = {1, -1, 2, -2};
	double k2[] = {1, 1, 1, 1};
	vector<double> retVal(8);
	for (int bb = 0; bb < 4; bb++) {
		double p1 = 1 / ((1 / p) + k1[bb] * (1 / (k2[bb] * p0)));
		retVal.push_back(p1);
		retVal.push_back(2 * p1);
	}
	return retVal;
}

double D2::Compute2DSpectrum(const string& outputFilePrefix) {
	double d2Freq = 0;

	if (procId == 0) {
		cout << "dmin = " << dmin << endl;
		cout << "dmax = " << dmax << endl;
		cout << "wmin = " << wmin << endl;
		cout << "numFreqs = " << numFreqs << endl;
		cout << "freqStep = " << freqStep << endl;
		cout << "numCoherenceBins = " << numCoherenceBins << endl;
		cout << "numCoherences = " << numCoherences << endl;
		cout << "a = " << a << endl;
		cout << "b = " << b << endl;
		cout << "coherenceBinSize = " << coherenceBinSize << endl;

		ofstream output(outputFilePrefix + ".csv");
		//ofstream output_min(outputFilePrefix + "_min.csv");
		//ofstream output_max(outputFilePrefix + "_max.csv");
		ofstream output_per(outputFilePrefix + "_per.csv");
		output_per << "f p d2" << endl;

		vector<double> specInt;
		double intMax = -1;
		double intMin = -1;
		// Basic cycle with printing for GnuPlot
		double deltac = maxCoherence > minCoherence ? (maxCoherence - minCoherence) / (numCoherences - 1) : 0;
		for (unsigned i = 0; i < numCoherences; i++) {
			double d = minCoherence + i * deltac;
			vector<pair<double, double>> spec(numFreqs);
			bool maxXReached = false;
			for (unsigned j = 0; j < numFreqs; j++) {
				double w = wmin + j * freqStep;
				double d1 = d;
				if (relative) {
					d1 = d / w;
				}
				if (d1 > maxX) {
					maxXReached = true;
					break;
				}
				double res = Criterion(d1, w);
				spec[j] = {w, res};
			}
			if (maxXReached) {
				break;
			}

			// Normalize locally if needed
			if (normalize) {
				Normalize(spec);
			}

			//ofstream output_mid("phasedisp" + to_string(i) + ".csv");
			double integral = 0;
			for (auto& s : spec) {
				double w = s.first;
				output << d << " " << w << " " << s.second << endl;
				if (i == 0) {
					//output_min << w << " " << s.second << endl;
				} else if (i == numCoherences - 1) {
					//output_max << w << " " << s.second << endl;
				}
				//output_mid << (wmin + j * step) << " " << spec[j] << endl;
				integral += s.second;
			}
			specInt.push_back(integral);
			if (intMax < 0 || integral > intMax) {
				intMax = integral;
			}
			if (intMin < 0 || integral < intMin) {
				intMin = integral;
			}
			if (i == 0) {
				vector<pair<double, double>> minima = getLocalMinima(spec);
				while (minima.size() > 10) {
					minima = getLocalMinima(minima);
				}
				if (removeSpurious) {
					for (auto i = minima.begin(); i != minima.end();) {
						cout << "Checking minimum: " << (1 / (*i).first) << endl;
						bool spurious = false;
						for (auto& m : minima) {
							if (m.first > (*i).first) {
								for (auto s : getSpurious(1 / m.first)) {
									if (abs(1/s - (*i).first) < 2 * freqStep) {
										cout << "Period " << (1 / (*i).first) << " is spurious of " << 1 / m.first << endl;
										spurious = true;
										break;
									}
								}
							}
						}
						if (spurious) {
							minima.erase(i);
						} else {
							i++;
						}
					}
				}
				if (!minima.empty()) {
					pair<double, double> minimum = minima[0];
					for (auto& m : minima) {
						output_per << m.first << " " << 1 / m.first << " " << m.second << endl;
						if (m.second < minimum.second) {
							minimum = m;
						}
					}
					d2Freq = minimum.first;
				}
			}
			//cout << endl;
			//output_mid.close();
		}
		output.close();
		//output_min.close();
		//output_max.close();
		output_per.close();

		// print normalized integral
		/*
		ofstream output_stats("stats.csv");
		//assert(intMin >=0 && intMax >= 0 && intMin < intMax);
		//double norm = 1 / (intMax - intMin);
		for (unsigned i = 0; i < specInt.size(); i++) {
			double d = minCoherence + i * deltac;
			output_stats << d << " " << specInt[i] / intMin << " " << specMinima[i] / minimaMin  << " " << (i > 0 ? (specMinima[i] - specMinima[i - 1]) : (specMinima[1] - specMinima[0])) << endl;
		}
		output_stats.close();
		*/
	}
	return d2Freq;
}

