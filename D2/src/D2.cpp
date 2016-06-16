#define BOOST_NO_CXX11_SCOPED_ENUMS

#include "D2.h"
#include "common.h"

#include <utility>
#include <limits>
#include <iostream>
#include <string>
#include <cmath>
#include <math.h>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;

#define square(x) ((x) * (x))

D2::D2(DataLoader* pDataLoader,
		double duration,
		double minPeriod, double maxPeriod,
		double minCoherence, double maxCoherence, int numFreqs,
		Mode mode, bool normalize, bool relative,
		double tScale, double startTime,
		const vector<double>& varScales, const vector<pair<double, double>>& varRanges,
		bool removeSpurious, int bootstrapSize, bool saveDiffNorms, bool saveParameters) :
			mpDataLoader(pDataLoader),
			minCoherence(minCoherence),
			maxCoherence(maxCoherence),
			numFreqs(numFreqs),
			mode(mode),
			normalize(normalize),
			relative(relative),
			tScale(tScale),
			startTime(startTime),
			varScales(varScales),
			varRanges(varRanges),
			removeSpurious(removeSpurious),
			bootstrapSize(bootstrapSize),
			saveDiffNorms(saveDiffNorms),
			saveParameters(saveParameters),
			e1(rd()) {
	if (pDataLoader) {
		assert(varScales.size() == pDataLoader->GetVarIndices().size());
	}

	wmax = 1.0 / minPeriod;
	wmin = 1.0 / maxPeriod;

	dmin = minCoherence * (relative ? minPeriod : 1);
	dmax = maxCoherence * (relative ? maxPeriod : 1);
	if (duration > 0 && dmax > duration * tScale) {
		dmax = duration * tScale;
		this->maxCoherence = floor(duration * tScale / maxPeriod);
	}
	dbase = dmin / 10;
	dmaxUnscaled = dmax / tScale;

	if (dmax < dmin || this->maxCoherence < this->minCoherence) {
		throw "Maximum coherence length smaller than minimum coherence length";
	}
	if (minPeriod > maxPeriod) {
		throw "Minimum period greater than maximum period";
	}
	if (duration > 0) {
		endTime = startTime + duration;
	} else {
		endTime = numeric_limits<double>::max();
	}

	numCoherences = dmax > dmin ? coherenceGrid : 1; // output precision in coherence
	numCoherenceBins = round(phaseBins * (dmax - dbase) * wmax);
	coherenceBinSize = (dmax - dbase) / (numCoherenceBins - 1);

	freqStep = (wmax - wmin) / (numFreqs - 1);
	eps = epsilon;
	epslim = 1.0 - eps;
	ln2 = sqrt(log(2.0));
	lnp = ln2 / eps;
}

pair<double, double> D2::Criterion(int bootstrapIndex, double d, double w) {
	double tyv = 0;
	int tav = 0;
	switch (mode) {
	case Box:
		#ifdef _OPENMP
			#pragma omp parallel for reduction(+:tyv,tav)
		#endif
		for (unsigned j = 0; j < td[bootstrapIndex].size(); j++) {
			double dd = td[bootstrapIndex][j];
			if (dd <= d) {
				double ph = dd * w - floor(dd * w);
				if (ph < 0) {
					ph = ph + 1;
				}
				if (ph < eps || ph > epslim) {
					tyv += ty[bootstrapIndex][j];
					tav += ta[bootstrapIndex][j];
				}
			}
		}
		break;
	case Gauss:
	case GaussCosine:
		#ifdef _OPENMP
			#pragma omp parallel for reduction(+:tyv,tav)
		#endif
		for (unsigned j = 0; j < td[bootstrapIndex].size(); j++) {// to jj-1 do begin
			double dd = td[bootstrapIndex][j];
			double ww;
			if (d > 0.0) {
				ww = exp(-square(ln2 * dd / d));
			} else {
				ww = 0.0;
			}
			double ph = dd * w - floor(dd * w);
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
				const double df = 0.5; // With this value we have wp = 1 + 2 cos (2 pi w (ti - tj))
				if (ph >= df) {
					wp = 0;
				} else if (ph == 0) {
					wp = 1;
				} else {
					wp = 2 * cos(M_PI * ph / df) + 1;
				}
				if (std::isnan(wp)) {
					wp = 0;
					cout << "wp is still nan" << endl;
				}
			}
			if (closeInPhase) {
				tyv += ww * wp * ty[bootstrapIndex][j];
				tav += ww * wp * ta[bootstrapIndex][j];
			}
		}
		break;
	}
	return {tyv, tav};
}

void Normalize(vector<D2SpecLine>& spec) {

	unique_ptr<double> min;
	unique_ptr<double> max;

	for (auto& s : spec) {
		if (!min || s.value < *min) {
			min.reset(new double(s.value));
		}
		if (!max || s.value > *max) {
			max.reset(new double(s.value));
		}
	}
	if (max && min && *max > *min) {
		double range = *max - *min;
		for (auto& s : spec) {
			s.value = (s.value - *min) / range;
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



bool D2::ProcessPage(DataLoader& dl1, DataLoader& dl2, double* tty, int* tta) {
	if (dl2.GetX(0) - dl1.GetX(dl1.GetPageSize() - 1) > dmaxUnscaled) {
		return false;
	}
	unsigned bsIndexes1[bootstrapSize][dl1.GetPageSize()];
	unsigned bsIndexes2[bootstrapSize][dl2.GetPageSize()];
	if (GetProcId() == 0) {
		uniform_int_distribution<unsigned> uniform_dist1(0, dl1.GetPageSize() - 1);
		uniform_int_distribution<unsigned> uniform_dist2(0, dl2.GetPageSize() - 1);
		for (auto bootstrapIndex = 0; bootstrapIndex < bootstrapSize; bootstrapIndex++) {
			for (unsigned i = 0; i < dl1.GetPageSize(); i++) {
				bsIndexes1[bootstrapIndex][i] = uniform_dist1(e1);
			}
			for (unsigned j = 0; j < dl2.GetPageSize(); j++) {
				bsIndexes2[bootstrapIndex][j] = uniform_dist2(e1);
			}
		}
#ifndef _NOMPI
		for (auto procNo = 1; procNo < GetNumProc(); procNo++) {
			MPI::COMM_WORLD.Send(bsIndexes1, bootstrapSize * dl1.GetPageSize(), MPI::INT, procNo, TAG_IND1);
		}
		recvLog();
		for (auto procNo = 1; procNo < GetNumProc(); procNo++) {
			MPI::COMM_WORLD.Send(bsIndexes2, bootstrapSize * dl2.GetPageSize(), MPI::INT, procNo, TAG_IND2);
		}
		recvLog();
#endif
	} else {
#ifndef _NOMPI
		MPI::Status status;
		MPI::COMM_WORLD.Recv(bsIndexes1, bootstrapSize * dl1.GetPageSize(),  MPI::INT, 0, TAG_IND1, status);
		assert(status.Get_error() == MPI::SUCCESS);
		sendLog("Received bootstrap indexes for 1.\n");
		MPI::COMM_WORLD.Recv(bsIndexes2, bootstrapSize * dl2.GetPageSize(),  MPI::INT, 0, TAG_IND2, status);
		assert(status.Get_error() == MPI::SUCCESS);
		sendLog("Received bootstrap indexes for 2.\n");
#endif
	}
	for (auto bootstrapIndex = 0; bootstrapIndex < bootstrapSize + 1; bootstrapIndex++) {
		//cout << "bootstrapIndex: " << bootstrapIndex << endl;
		for (unsigned i = 0; i < dl1.GetPageSize(); i++) {
			unsigned j = 0;
			if (bootstrapIndex == 0 && dl1.GetPage() == dl2.GetPage()) {
				j = i + 1;
			}
			//if (GetProcId() == 0) {
			//	cout << "Time :" << dl1.GetX(i) << endl;
			//}
			for (; j < dl2.GetPageSize(); j++) {
				auto ix = i;
				auto jx = j;
				auto iy = i;
				auto jy = j;
				if (bootstrapIndex > 0) {
					ix = bsIndexes1[bootstrapIndex - 1][i];
					jx = bsIndexes2[bootstrapIndex - 1][j];
					if (confIntOrSignificance) {
						iy = ix;
						jy = jx;
					}
				}
				real xiUnscaled = dl1.GetX(ix);
				real xjUnscaled = dl2.GetX(jx);
				real xi = xiUnscaled * tScale;
				real xj = xjUnscaled * tScale;
				//if (bootstrapIndex == 0) {
				//	if (xj > maxX) {
				//		maxX = xj;
				//	}
				//}
				real d = xj - xi;
				if (bootstrapSize == 0 && (d > dmax || xjUnscaled > endTime)) {
					//cout << "Breaking GetProcId, i, d: " << GetProcId() << ", " << bootstrapIndex << ", " << d << endl;
					break;
				}
				if (xiUnscaled >= startTime && (d >= dbase && d <= dmax && xjUnscaled <= endTime)) {
					//numCoherenceBins = round(phaseBins * (dmax - dbase) * wmax);
					//a = (numCoherenceBins - 1.0) / (dmax - dbase);
					//b = -dbase * a;
					//int kk = round(a * d + b);
					int kk = round((d - dbase) * (numCoherenceBins - 1) / (dmax - dbase));
					//cout << "GetProcId, i, d, kk: " << GetProcId() << ", " << bootstrapIndex << ", " << d << ", " << kk << endl;
					auto dy2 = DiffNorm(dl2.GetY(jy), dl1.GetY(iy));
					tty[bootstrapIndex * numCoherenceBins + kk] += dy2;
					tta[bootstrapIndex * numCoherenceBins + kk]++;
					//cout << "tta[" << kk << "]=" << tta[bootstrapIndex][kk] << endl;
					//cout << "tty[" << kk << "]=" << tty[kk] << endl;
				}
			}
		}
	}
	return true;
}

void D2::VarCalculation(double* ySum, double* y2Sum) {
	for (unsigned i = 0; i < mpDataLoader->GetPageSize(); i++) {
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
}



void D2::CalcDiffNorms() {
	assert(mpDataLoader); // dataLoader must be present in case diffnorms are not calculated yet
	sendLog("Calculating diffnorms...\n");
	recvLog();

	sendLog(string("numCoherenceBins: ") + to_string(numCoherenceBins) + "\n");
	recvLog();
	// Allocate dynamically (may not fit into stack)
	double* tty = new double[(bootstrapSize + 1) * numCoherenceBins];
	int* tta = new int[(bootstrapSize + 1) * numCoherenceBins];
	for (auto bootstrapIndex = 0; bootstrapIndex < bootstrapSize + 1; bootstrapIndex++) {
		for (unsigned i = 0; i < numCoherenceBins; i++) {
			tty[bootstrapIndex * numCoherenceBins + i] = 0;
			tta[bootstrapIndex * numCoherenceBins + i] = 0;
		}
	}


	// Now comes precomputation of differences and counts. They are accumulated in two grids.
	sendLog("Loading data...\n");
	recvLog();

	unsigned size = mpDataLoader->GetNumVars() * mpDataLoader->GetDim();
	double ySum[size];
	double y2Sum[size];
	int n = 0;
	for (unsigned i = 0; i < size; i++) {
		ySum[i] = 0;
		y2Sum[i] = 0;
	}
	sendLog(string("dbase, dmax, numCoherenceBins: ") + to_string(dbase) + ", " + to_string(dmax) + ", " + to_string(numCoherenceBins) + "\n");
	recvLog();
	while (mpDataLoader->Next()) {
		n += mpDataLoader->GetPageSize();
		VarCalculation(ySum, y2Sum);
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
		sendLog(string("Page ") + to_string(mpDataLoader->GetPage()) + " loaded.\n");
		recvLog();
		delete dl2;
	}
	varSum = 0;
	for (unsigned i = 0; i < size; i++) {
		varSum += (y2Sum[i] - (ySum[i] * ySum[i]) / n) / (n - 1);
	}
	if (GetProcId() == 0) {
		cout << "Waiting for data from other processes..." << endl;
	}
#ifndef _NOMPI
	MPI::COMM_WORLD.Barrier();
#endif
	if (GetProcId() > 0) {
		sendLog(string("Sending data from ") + to_string(GetProcId()) + ".\n");
		//for (unsigned j = 0; j < m; j++) {
		//	cout << tty[j] << endl;
		//}
#ifndef _NOMPI
		MPI::COMM_WORLD.Send(tty, (bootstrapSize + 1) * numCoherenceBins, MPI::DOUBLE, 0, TAG_TTY);
		MPI::COMM_WORLD.Send(tta, (bootstrapSize + 1) * numCoherenceBins, MPI::INT, 0, TAG_TTA);
		MPI::COMM_WORLD.Send(&varSum, 1, MPI::DOUBLE, 0, TAG_VAR);
#endif
	} else {
#ifndef _NOMPI
		recvLog();
		for (int i = 1; i < GetNumProc(); i++) {
			double* ttyRecv = new double[(bootstrapSize + 1) * numCoherenceBins];
			int* ttaRecv = new int[(bootstrapSize + 1) * numCoherenceBins];
			MPI::Status status;
			MPI::COMM_WORLD.Recv(ttyRecv, (bootstrapSize + 1) * numCoherenceBins,  MPI::DOUBLE, MPI_ANY_SOURCE, TAG_TTY, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received square differences from " << status.Get_source() << "." << endl;
			MPI::COMM_WORLD.Recv(ttaRecv, (bootstrapSize + 1) * numCoherenceBins,  MPI::INT, status.Get_source(), TAG_TTA, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received weights from " << status.Get_source() << "." << endl;
			for (auto bootstrapIndex = 0; bootstrapIndex < bootstrapSize + 1; bootstrapIndex++) {
				for (unsigned j = 0; j < numCoherenceBins; j++) {
					tty[bootstrapIndex * numCoherenceBins + j] += ttyRecv[bootstrapIndex * numCoherenceBins + j];
					//cout << "Weigths: " << bootstrapIndex << " " << tta[bootstrapIndex][j] << " " << ttaRecv[bootstrapIndex][j] << endl;
					if (tta[bootstrapIndex * numCoherenceBins + j] != ttaRecv[bootstrapIndex * numCoherenceBins + j]) {
						cout << "tta discrepancy: " << i << " " << bootstrapIndex << " " << j << " " << tta[bootstrapIndex * numCoherenceBins + j] << " " << ttaRecv[bootstrapIndex * numCoherenceBins + j] << endl;
					}
					//assert(tta[bootstrapIndex * numCoherenceBins + j] == ttaRecv[bootstrapIndex * numCoherenceBins + j]);
				}
			}
			double varSumRecv;
			MPI::COMM_WORLD.Recv(&varSumRecv, 1,  MPI::DOUBLE, status.Get_source(), TAG_VAR, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "Received variance sum " << status.Get_source() + "." << endl;
			varSum += varSumRecv;
		}
#endif
		cout << "varSum: " << varSum << endl;
		ta.resize(bootstrapSize + 1);
		ty.resize(bootstrapSize + 1);
		td.resize(bootstrapSize + 1);
		for (auto bootstrapIndex = 0; bootstrapIndex < bootstrapSize + 1; bootstrapIndex++) {
			// How many time differences was actually used?
			unsigned j = 0;
			for (unsigned i = 0; i < numCoherenceBins; i++) {
				if (tta[bootstrapIndex * numCoherenceBins + i] > 0) {
					j++;
				}
			}
			//cout << "j=" << j << endl;
			ta[bootstrapIndex].assign(j, 0);
			ty[bootstrapIndex].assign(j, 0);
			td[bootstrapIndex].assign(j, 0);
			//cout << "td.size()=" << td[bootstrapIndex].size() << endl;

			// Build final grids for periodicity search.

			j = 0;
			for (unsigned i = 0; i < numCoherenceBins; i++) {
				double d = dbase + i * coherenceBinSize;
				if (tta[bootstrapIndex * numCoherenceBins + i] > 0) {
					td[bootstrapIndex][j] = d;
					ty[bootstrapIndex][j] = tty[bootstrapIndex * numCoherenceBins + i];
					ta[bootstrapIndex][j] = tta[bootstrapIndex * numCoherenceBins + i];
					j++;
				}
			}
			if (bootstrapIndex == 0 && saveDiffNorms) {
				string diffNormsFilePrefix = string(DIFF_NORMS_FILE_PREFIX);
				ofstream output(diffNormsFilePrefix + "_" + to_string(GetCurrentTime()) + DIFF_NORMS_FILE_SUFFIX);
				output << varSum << endl;
				for (unsigned i = 0; i < j; i++) {
					output << td[0][i] << " " << ty[0][i] << " " << ta[0][i] << endl;
				}
				output.close();
			}
		}

		if (saveParameters) {
			copy_file(GetParamFileName(), string(PARAMETERS_FILE_PREFIX) + "_" + to_string(GetCurrentTime()) + PARAMETERS_FILE_SUFFIX);
		}
	}
	delete[] tty;
	delete[] tta;
}

void D2::LoadDiffNorms() {
	if (GetProcId() == 0) {
		td.resize(1);
		ty.resize(1);
		ta.resize(1);
		varSum = 1; // Assuming unit variance by default
		cout << "Loading diffnorms..." << endl;
		string diffNormsFile = DIFF_NORMS_FILE;
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
					td[0].push_back(stod(words[0]));
					ty[0].push_back(stod(words[1]));
					ta[0].push_back(stoi(words[2]));
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

vector<D2SpecLine> getLocalMinima(const vector<D2SpecLine>& spec, int maxCount) {
	int minSeparation = spec.size() / (2 * maxCount);
	//cout << "minSeparation: " << minSeparation << endl;
	vector<D2SpecLine> retVal;
	vector<int> usedIndices;
	for (int count = 0; count < maxCount; count++) {
		D2SpecLine globalMinimum(0, numeric_limits<double>::max(), 1, 1);
		int minimumIndex = 0;
		for (int i = 1; i < ((int) spec.size()) - 1; i++) {
			bool tooClose = false;
			for (auto usedIndex : usedIndices) {
				if (abs (i - usedIndex) <= minSeparation) {
					tooClose = true;
					break;
				}
			}
			if (!tooClose) {
				if (spec[i].value < globalMinimum.value) {
					globalMinimum = spec[i];
					minimumIndex = i;
				}
			}
		}
		if (minimumIndex > 0) {
			retVal.push_back(globalMinimum);
			usedIndices.push_back(minimumIndex);
		} else {
			break;
		}
	}
	return retVal;
}

pair<double, double> confInt(const D2SpecLine& minimum, const vector<D2SpecLine>& spec) {
	//cout << "Conf. int. for minimum: " << minimum.frequency << " " << minimum.tyv << " " << minimum.tav << endl;
	size_t index = 0;
	for (auto s : spec) {
		if (s.frequency == minimum.frequency) {
			break;
		}
		index++;
	}
	double lower = -1;
	double upper = -1;
	if (index < spec.size()) {
		for (int i = index - 1; i >= 0; i--) {
			double probFact = exp(minimum.tav * (1 - minimum.tav * spec[i].tyv / minimum.tyv / spec[i].tav));
			//cout << "Lower: " << spec[i].frequency << " " << spec[i].tyv << " " << spec[i].tav << " " << probFact << endl;
			if (probFact < 0.05) {
				lower = spec[i].frequency;
				break;
			}
		}
		for (size_t i = index + 1; i < spec.size(); i++) {
			double probFact = exp(minimum.tav * (1 - minimum.tav * spec[i].tyv / minimum.tyv / spec[i].tav));
			//cout << "Upper: " << spec[i].frequency << " " << spec[i].tyv << " " << spec[i].tav << " " << probFact << endl;
			if (probFact < 0.05) {
				upper = spec[i].frequency;
				break;
			}
		}
	}
	return {lower, upper};
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

double calcEntropy(const vector<pair<double, double>>& spec) {
    size_t n = spec.size();
	double norm = 0;
    for (auto& s : spec) {
    	norm += s.second + 1e-12;
    }
    double entropy = 0;
    for (auto& s : spec) {
    	double d = s.second / norm;
    	entropy -= d * log2(d + 1e-12);
    }
    entropy /= log2(n);
    return entropy;
}

pair<double /*mean*/, double /*variance*/> meanVariance(const vector<D2SpecLine>& spec) {
	assert(spec.size() > 0);
	double sum = 0;
	double sumSquares = 0;
	for(auto& s : spec) {
		sum += s.value;
		sumSquares += s.value * s.value;
	}
	int n = spec.size();
	double mean = sum / n;
	return {mean,  sumSquares / n - mean * mean};
}

double calcBimodality(const vector<D2SpecLine>& spec) {
    size_t n = spec.size();
    assert(n > 3);
    auto mv = meanVariance(spec);
    auto mean = mv.first;
    auto var = mv.second;
    double skewness = 0;
    double excessKurtosity = 0;
    for (auto& s : spec) {
    	double diff = s.value - mean;
    	double diffSqr = diff * diff;
    	skewness += diff * diffSqr;
    	excessKurtosity += diffSqr * diffSqr;
    }
    skewness /= n * var * sqrt(var);
    excessKurtosity /= n * (var * var);
    excessKurtosity -= 3;

    return (skewness * skewness + 1) / (excessKurtosity + 3 * (n - 1) * (n - 1) / (n - 2) / (n - 3));

}

void D2::RemoveSpurious(vector<D2SpecLine>& minima) {
	for (auto i = minima.begin(); i != minima.end();) {
		cout << "Checking minimum: " << (1 / (*i).frequency) << endl;
		bool spurious = false;
		for (auto& m : minima) {
			if (m.frequency > (*i).frequency) {
				for (auto s : getSpurious(1 / m.frequency)) {
					if (abs(1/s - (*i).frequency) < 2 * freqStep) {
						cout << "Period " << 1 / (*i).frequency << " is spurious of " << 1 / m.frequency << endl;
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

//TODO: printing the results to file should be taken out of from this method
const vector<D2Minimum> D2::Compute2DSpectrum(int bootstrapIndex, const string& outputFilePrefix) {
	ofstream output;
	ofstream output_min;
	ofstream output_max;
	if (bootstrapIndex == 0) {
		cout << "dmin = " << dmin << endl;
		cout << "dmax = " << dmax << endl;
		cout << "wmin = " << wmin << endl;
		cout << "wmax = " << wmax << endl;
		cout << "numFreqs = " << numFreqs << endl;
		cout << "freqStep = " << freqStep << endl;
		cout << "numCoherenceBins = " << numCoherenceBins << endl;
		cout << "numCoherences = " << numCoherences << endl;
		cout << "coherenceBinSize = " << coherenceBinSize << endl;

		output.open(outputFilePrefix + ".csv");
		output_min.open(outputFilePrefix + "_min.csv");
		output_max.open(outputFilePrefix + "_max.csv");
	}

	vector<D2Minimum> allMinima;
	vector<double> bimodality;
	//vector<double> specInt;
	//double intMax = -1;
	//double intMin = -1;

	double deltac = maxCoherence > minCoherence ? (maxCoherence - minCoherence) / (numCoherences - 1) : 0;
	vector<D2SpecLine> prevSpec(numFreqs);
	for (unsigned i = 0; i < numCoherences; i++) {
		double d = minCoherence + i * deltac;
		vector<D2SpecLine> spec(numFreqs);
		bool maxXReached = false;
		for (int j = 0; j < numFreqs; j++) {
			double w = wmin + j * freqStep;
			double d1 = d;
			if (relative) {
				d1 = d / w;
			}
			//if (d1 > maxX) {
			//	maxXReached = true;
			//	break;
			//}
			auto res = Criterion(bootstrapIndex, d1, w);
			double tyv = res.first;
			double tav = res.second;
			if (tav > 0 && varSum > 0) {
				spec[j] = D2SpecLine(w, tyv, tav, varSum);
			} else {
				cout << "tav=" << tav << endl;
				// Commented out, because default constructor of D2SpecLine
				// is already called when the spec vector is constructed
				// spec[j] = D2SpecLine();
			}
		}
		if (maxXReached) {
			break;
		}

		vector<D2SpecLine> specCopy = spec;
		if (differential && i > 0) {
			for (int j = 0; j < numFreqs; j++) {
				auto& s = spec[j];
				if (differential) {
					s.value -= prevSpec[j].value;
				}
			}
		}
		// Normalize locally if needed
		if (normalize) {
			Normalize(spec);
		}

		if (bootstrapIndex == 0) {
			if (!differential || i > 0) {
				//ofstream output_mid("phasedisp" + to_string(i) + ".csv");
				//double integral = 0;
				for (auto& s : spec) {
					double w = s.frequency;
					auto d2 = s.value;
					output << d << " " << w << " " << d2 << endl;
					if (i == 0) {
						output_min << w << " " << d2 << endl;
					} else if (i == numCoherences - 1) {
						output_max << w << " " << d2 << endl;
					}
					//output_mid << (wmin + j * step) << " " << spec[j] << endl;
					//integral += s.second;
				}
				bimodality.push_back(calcBimodality(spec));
				//specInt.push_back(integral);
				//if (intMax < 0 || integral > intMax) {
				//	intMax = integral;
				//}
				//if (intMin < 0 || integral < intMin) {
				//	intMin = integral;
				//}
			}
		}
		prevSpec = specCopy;
		if (!differential) {
			double log2i = log2(i);
			if (floor(log2i) == log2i) {
				// 1 minimum for shortest coherence length, more for longer ones
				auto minima = getLocalMinima(spec, round(d / minCoherence * wmax / wmin) );
				if (removeSpurious) {
					RemoveSpurious(minima);
				}
				for (auto& m : minima) {
					auto ci = confInt(m, spec);
					if (ci.first > 0 && ci.second > 0) {
						allMinima.push_back(D2Minimum(d, m.frequency, m.value, ci));
						if (m.frequency - ci.first <= freqStep || ci.second - m.frequency <= freqStep) {
							cout << "Possibly too wide confidence interval estimated for minimum at d=" << d << ", f=" <<  m.frequency << endl;
						}
					} else {
						cout << "Skipping insignificant minimum at d=" << d << ", f=" <<  m.frequency << endl;
					}
				}
			}
		}
		//cout << endl;
		//output_mid.close();
	}
	if (bootstrapIndex == 0) {
		output.close();
		output_min.close();
		output_max.close();

		// print some other stats

		ofstream output_stats("stats.csv");
		for (unsigned i = 0; i < bimodality.size(); i++) {
			double d = minCoherence + i * deltac;
			output_stats << d << " " << bimodality[i] << endl;
		}
		output_stats.close();
		this->allMinima = allMinima;
	}
	return allMinima;
}

void D2::Bootstrap(const string& outputFilePrefix) {
	if (GetProcId() == 0) {
		ofstream output_bootstrap(outputFilePrefix + "_bootstrap.csv");
		for (auto i = 0; i < bootstrapSize; i++) {
				auto minima = Compute2DSpectrum(i + 1, "");
				for (auto& m : minima) {
					output_bootstrap << m.coherenceLength << " " << m.frequency << " " << 1 / m.frequency << " " << m.value << endl;
				}
				output_bootstrap << endl;
		}
		output_bootstrap.close();
	}
}
