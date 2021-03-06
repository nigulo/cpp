/*
 * main.cpp
 *
 *  Created on: May 4, 2016
 *      Author: nigul
 */


#include "D2.h"
#include "BinaryDataLoader.h"
#include "TextDataLoader.h"
#include "utils/utils.h"
#include "common.h"

#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <sstream>
#include <iomanip>

using namespace utils;
using namespace std;
using namespace boost;
using namespace boost::filesystem;

int procId;
int numProc;
time_t currentTime;

string paramFileName;

int GetProcId() {
	return procId;
}

int GetNumProc() {
	return numProc;
}

time_t GetCurrentTime() {
	return currentTime;
}

const string& GetParamFileName() {
	return paramFileName;
}

void sendLog(const string& str) {
	if (GetProcId() > 0) {
#ifndef _NOMPI
		int len = str.length() + 1;
		MPI::COMM_WORLD.Send(&len, 1, MPI::INT, 0, TAG_LOG_LEN);
		MPI::COMM_WORLD.Send(str.c_str(), len, MPI::CHAR, 0, TAG_LOG);
		MPI::COMM_WORLD.Barrier();
#endif
	} else {
		cout << "PROC" << GetProcId() << ": " << str;
	}
}

void recvLog() {
#ifndef _NOMPI
	if (GetProcId() == 0) {
		for (int i = 1; i < GetNumProc(); i++) {
			int len;
			MPI::Status status;
			MPI::COMM_WORLD.Recv(&len, 1,  MPI::INT, MPI_ANY_SOURCE, TAG_LOG_LEN, status);
			assert(status.Get_error() == MPI::SUCCESS);
			char strRecv[len];
			int source = status.Get_source();
			MPI::COMM_WORLD.Recv(strRecv, len,  MPI::CHAR, source, TAG_LOG, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "PROC" << source << ": " << strRecv;
		}
		MPI::COMM_WORLD.Barrier();
	}
#endif
}

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

template<> string vecToStr<pair<int, int>>(const vector<pair<int, int>>& vec) {
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
			cout << "Usage: ./D2 [param file] [params to overwrite]\nparam file defaults to " << PARAMETERS_FILE << endl;
			return EXIT_SUCCESS;
		}
	}
	currentTime = time(nullptr);
	paramFileName = argc > 1 ? argv[1] : PARAMETERS_FILE;
	string cmdLineParams = argc > 2 ? argv[2] : "";
	boost::replace_all(cmdLineParams, " ", "\n");

	if (procId == 0) {
		if (!exists(paramFileName)) {
			cout << "Cannot find " << paramFileName << endl;
			return EXIT_FAILURE;
		}
	}
	map<string, string> paramsFromFile = Utils::ReadProperties(paramFileName);
	map<string, string> params = Utils::ReadPropertiesFromString(cmdLineParams);
	for(const auto& entry : paramsFromFile) {
		params.insert({entry.first, entry.second});
	}
	double duration = Utils::FindDoubleProperty(params, "duration", 0);
	double initMinPeriod = Utils::FindDoubleProperty(params, "minPeriod", 2);
	double initMaxPeriod = Utils::FindDoubleProperty(params, "maxPeriod", 10);
	double minCoherence = Utils::FindDoubleProperty(params, "minCoherence", 3);
	double maxCoherence = Utils::FindDoubleProperty(params, "maxCoherence", 0);
	int bootstrapSize = Utils::FindIntProperty(params, "bootstrapSize", 0);
	assert(bootstrapSize >= 0);
	int numFreqs = Utils::FindIntProperty(params, "numFreqs", bootstrapSize > 0 ? max(1000, 100000 / bootstrapSize) : 100000);

	string phaseSelFnStr = Utils::FindProperty(params, "phaseSelFn", "Cosine");
	to_upper(phaseSelFnStr);
	PhaseSelFn phaseSelFn;
	if (phaseSelFnStr == "BOX") {
		phaseSelFn = PhaseSelFnBox;
	} else if (phaseSelFnStr == "GAUSS") {
		phaseSelFn = PhaseSelFnGauss;
	} else if (phaseSelFnStr == "COSINE") {
		phaseSelFn = PhaseSelFnCosine;
	} else {
		cerr << "Invalid phaseSelFn" << endl;
		assert(false);
	}

	string timeSelFnStr = Utils::FindProperty(params, "timeSelFn", "Gauss");
	to_upper(timeSelFnStr);
	TimeSelFn timeSelFn;
	if (timeSelFnStr == "NONE") {
		timeSelFn = TimeSelFnNone;
	} else if (timeSelFnStr == "BOX") {
		timeSelFn = TimeSelFnBox;
	} else if (timeSelFnStr == "GAUSS") {
		timeSelFn = TimeSelFnGauss;
	} else {
		cerr << "Invalid timeSelFn" << endl;
		assert(false);
	}
	//-------------------------------------------------------------
	// For backward compatibility
	string modeStr = Utils::FindProperty(params, "mode", "");
	if (modeStr.length() > 0) {
		to_upper(modeStr);
		if (modeStr == "BOX") {
			phaseSelFn = PhaseSelFnBox;
			timeSelFn = TimeSelFnBox;
		} else if (modeStr == "GAUSS") {
			phaseSelFn = PhaseSelFnGauss;
			timeSelFn = TimeSelFnGauss;
		} else if (modeStr == "GAUSSCOSINE" ||modeStr == "GAUSSWITHCOSINE") {
			phaseSelFn = PhaseSelFnCosine;
			timeSelFn = TimeSelFnGauss;
		} else {
			cerr << "Invalid mode" << endl;
			assert(false);
		}
	}
	//-------------------------------------------------------------
	bool normalize = Utils::FindIntProperty(params, "normalize", 0);
	bool relative = Utils::FindIntProperty(params, "relative", 1);
	bool differential = Utils::FindIntProperty(params, "differential", false);
	bool removeSpurious = Utils::FindIntProperty(params, "removeSpurious", 0);

	double tScale = Utils::FindDoubleProperty(params, "tScale", 1);
	int startTime = Utils::FindIntProperty(params, "startTime", 1);

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
			double min = minAndMax[0].empty() ? numeric_limits<double>::lowest() : stod(minAndMax[0]);
			double max = minAndMax[1].empty() ? numeric_limits<double>::max() : stod(minAndMax[1]);
			varRanges.push_back({min, max});
		}
	}

	// Determines whether bootstrap resampling is done for the purpose of
	// calculating confidence intervals or significance
	bool confIntOrSignificance = Utils::FindIntProperty(params, "confIntOrSignificance", 1);

	int numIterations = Utils::FindIntProperty(params, "numIterations", 1);
	if (numIterations > 1 && numProc > 1) {
		cout << "Zooming not available in multiprocessor regime" << endl;
		return EXIT_FAILURE;
	}
	double zoomFactor = Utils::FindDoubleProperty(params, "zoomFactor", 10);
	double smoothWindow = Utils::FindDoubleProperty(params, "smoothWindow", 0);

	assert(numIterations >= 1);
	assert(zoomFactor >= 1);

	if (procId == 0) {
		cout << "----------------" << endl;
		cout << "Parameter values" << endl;
		cout << "----------------" << endl;
		cout << "numProc        " << numProc << endl;
		cout << "bootstrapSize  " << bootstrapSize << endl;
		cout << "duration       " << duration << endl;
		cout << "minPeriod      " << initMinPeriod << endl;
		cout << "maxPeriod      " << initMaxPeriod << endl;
		if (timeSelFn != TimeSelFnNone) {
			cout << "minCoherence   " << minCoherence << endl;
		}
		if (timeSelFn != TimeSelFnNone && maxCoherence > 0) {
			cout << "maxCoherence   " << maxCoherence << endl;
		}
		cout << "numFreqs       " << numFreqs << endl;
		cout << "phaseSelFn     " << phaseSelFn << endl;
		cout << "timeSelFn      " << timeSelFn << endl;
		cout << "normalize      " << normalize << endl;
		cout << "relative       " << relative << endl;
		cout << "differential   " << differential << endl;
		cout << "removeSpurious " << removeSpurious << endl;
		cout << "tScale         " << tScale << endl;
		cout << "startTime      " << startTime << endl;
		cout << "varScales      " << vecToStr(varScales) << endl;
		cout << "numIterations  " << numIterations << endl;
		cout << "zoomFactor     " << zoomFactor << endl;
		cout << "smoothWindow   " << smoothWindow << endl;
		cout << "----------------" << endl;
	}

	string filePath;

	if (numProc == 1) {
		filePath = Utils::FindProperty(params, string("filePath"), "");
		if (filePath.empty()) {
			filePath = Utils::FindProperty(params, string("filePath0"), "");
		}
	} else {
		filePath = Utils::FindProperty(params, string("filePath") + to_string(procId), "");
	}
	string outputFilePrefix = Utils::FindProperty(params, string("outputFilePath"), "phasedisp");

	bool saveDiffNorms = Utils::FindIntProperty(params, "saveDiffNorms", 0);
	bool saveParameters = Utils::FindIntProperty(params, "saveParameters", 0);

	bool binary = Utils::FindIntProperty(params, "binary", 0);
	int bufferSize = Utils::FindIntProperty(params, "bufferSize", 0);

	string strDims = Utils::FindProperty(params, "dims", "1");
	vector<string> dimsStr;
	vector<int> dims;
	boost::split(dimsStr, strDims, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = dimsStr.begin() ; it != dimsStr.end(); ++it) {
		if ((*it).length() != 0) {
			dims.push_back(stoi(*it));
		}
	}

	string strNumProcs = Utils::FindProperty(params, "numProcs", "1");
	vector<string> numProcsStr;
	vector<int> numProcs;
	boost::split(numProcsStr, strNumProcs, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = numProcsStr.begin() ; it != numProcsStr.end(); ++it) {
		if ((*it).length() != 0) {
			numProcs.push_back(stoi(*it));
		}
	}

	assert(numProcs.size() == dims.size());
	assert(numProc == accumulate(numProcs.begin(), numProcs.end(), 1, multiplies<int>()));

	vector<int> dimsPerProc;
	vector<int> procIds;

	int procSize = 1;
	for (size_t i = 0; i < dims.size(); i++) {
		assert(dims[i] % numProcs[i] == 0);
		dimsPerProc.push_back(dims[i] / numProcs[i]);
		procIds.push_back((procId / procSize) % numProcs[i]);
		procSize *= numProcs[i];
		//cout << procId << ": procIds[" << i << "]" << "=" << procIds[i] << endl;
	}


	vector<vector<pair<int, int>>> regions;
	string strRegions = Utils::FindProperty(params, "regions", "");
	if (!strRegions.empty()) {
		for (string strRegion : Utils::SplitByChars(strRegions, ";")) {
			vector<pair<int, int>> region;
			int i = 0;
			for (string strMinMax : Utils::SplitByChars(strRegion, ",", false)) {
				vector<string> minMaxStrs = Utils::SplitByChars(strMinMax, ":-", false);
				assert(minMaxStrs.size() == 2);
				int min = stoi(minMaxStrs[0]);
				int procMin = procIds[i] * dimsPerProc[i];
				int max = stoi(minMaxStrs[1]);
				int procMax = (procIds[i] + 1) * dimsPerProc[i] - 1;
				//cout << procId << ": procMin, procMax [" << i << "] " << procMin << " " << procMax << endl;
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

	int totalNumVars = Utils::FindIntProperty(params, "numVars", 1);

	string strVarIndices = Utils::FindProperty(params, "varIndices", "0");
	vector<string> varIndicesStr;
	vector<int> varIndices;
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
			varRanges.push_back({numeric_limits<double>::lowest(), numeric_limits<double>::max()});
		}
	}

	assert(filePath.size() > 0);
	double minPeriod = initMinPeriod;
	double maxPeriod = initMaxPeriod;
	for (int i = 0; i < numIterations; i++) {
		DataLoader* dl = nullptr;
		if (exists(filePath)) {
			sendLog(string("Rank: ") + to_string(procId) + " file: " + filePath + "\n");
			recvLog();

			if (binary) {
				dl = new BinaryDataLoader(filePath, bufferSize, dimsPerProc, regions, totalNumVars, varIndices);
			} else {
				dl = new TextDataLoader(filePath, bufferSize, dimsPerProc, regions, totalNumVars, varIndices);
			}

		} else {
			cout << filePath << " does not exist" << endl;
		}
		if (timeSelFn == TimeSelFnNone || maxCoherence == 0) {
			auto rangeAndLen = dl->CalcRangeAndLength();
			cout << "dataRange     " << get<0>(rangeAndLen) << " - " << get<1>(rangeAndLen) << endl;
			cout << "dataLength     " << get<2>(rangeAndLen) << endl;
			auto timeRange = (get<1>(rangeAndLen) - get<0>(rangeAndLen)) * tScale;
			maxCoherence = timeRange;
			if (relative) {
				maxCoherence /= maxPeriod;
			}
			if (timeSelFn == TimeSelFnNone) {
				minCoherence = maxCoherence;
				cout << "minCoherence   " << minCoherence << endl;
			}
			cout << "maxCoherence   " << maxCoherence << endl;
		}

		D2 d2(dl, duration, minPeriod, maxPeriod, minCoherence, maxCoherence, numFreqs, phaseSelFn, timeSelFn,
				normalize, relative, tScale, startTime, varScales, varRanges, removeSpurious,
				bootstrapSize, saveDiffNorms, saveParameters);
		d2.confIntOrSignificance = confIntOrSignificance;
		d2.differential = differential;
		d2.smoothWindow = smoothWindow;
		if (!exists(DIFF_NORMS_FILE) || bootstrapSize > 0) {
			d2.CalcDiffNorms();
		} else {
			d2.LoadDiffNorms();
		}
		if (procId == 0) {
			auto& minima = d2.Compute2DSpectrum(0, outputFilePrefix);

			std::ofstream output_minima(outputFilePrefix + "_minima.csv");
			for (auto& m : minima) {
				output_minima
					<< std::setprecision(10) << m.coherenceLength << " "
					<< m.frequency << " "
					<< 1 / m.frequency << " "
					<< m.value << " "
					<< m.error << " "
					<< m.error / square(m.frequency) << endl;
			}
			output_minima.close();
			d2.Bootstrap(outputFilePrefix);

			if (dl) {
				delete dl;
			}
			// Zooming into the strongest minimum at smallest coherence length
			D2Minimum strongestMinimum(numeric_limits<double>::max(), 0, numeric_limits<double>::max(), 0);
			for (auto& m : minima) {
				if (m.coherenceLength < strongestMinimum.coherenceLength
						|| (m.coherenceLength < strongestMinimum.coherenceLength && m.value < strongestMinimum.value)) {
					strongestMinimum.frequency = m.frequency;
					strongestMinimum.value = m.value;
					strongestMinimum.coherenceLength = m.coherenceLength;
					strongestMinimum.error = m.error;
				}
			}
			if (i < numIterations - 1) {
				if (strongestMinimum.frequency == 0) {
					cout << "Finish zooming, no minima found" << endl;
					break;
				}
				double d2Period = 1 / strongestMinimum.frequency;
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
	}
#ifndef _NOMPI
		MPI::COMM_WORLD.Barrier();
#endif
	if (procId == 0) {
		cout << "done!" << endl;
	}
#ifndef _NOMPI
	MPI::Finalize();
#endif
	return EXIT_SUCCESS;
}
