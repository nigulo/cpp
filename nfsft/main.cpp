#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>

#include <fstream>
#include <vector>
#include "BinaryDataLoader.h"

#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include "utils/utils.h"
#include <iomanip>
#include "Transformer.h"

#define TEST
//#define ONLY_AZIMUTHAL_RECONST

using namespace utils;
using namespace std;
using namespace boost;
using namespace boost::filesystem;

string paramFileName;

#define RESULTS_OUT "resultsOut"
#define DATA_OUT "dataOut"
#define RECONST_OUT "reconstOut"

#define RESULTS_TXT "results.txt"
#define DATA_TXT "data.txt"
#define RECONST_TXT "reconst.txt"

void loadTestData(const map<string, string>& params) {
	cout << "Generating test data!" << endl;
    ofstream data_out(Utils::FindProperty(params, string(DATA_OUT), DATA_TXT));
	const int m_phi = 75;
	const int m_theta = 200;
	int N = m_phi;
	int M = m_phi * m_theta;
	Transformer transformer(N, M, Utils::FindProperty(params, RESULTS_OUT, RESULTS_TXT), Utils::FindProperty(params, RECONST_OUT, RECONST_TXT));
    transformer.init();
    /* define nodes and data*/
    int m = 0;
    double x1_step = 1.0 / m_phi;
    double pole_dist = 15;
    double x2_range = 0.5 * (180 - 2 * pole_dist) / 180;
    double x2_offset = 0.5 * pole_dist / 180;
    double x2_step = x2_range / m_theta;
    for (int i = 0; i < m_phi; i++) {
	    double x1 = -0.5 + x1_step * i;
	    for (int j = 0; j < m_theta; j++) {
		    double x2 = x2_offset + x2_step * j;
		    //plan.x[2*m] = x1;
		    //plan.x[2*m+1] = x2;
		    transformer.setX(m, x1, x2);
		    double field = /*sin(2*M_PI*(x1+0.5)) +*/ sin(2*2*M_PI*(x1+0.5)) * sin(3*4*M_PI*x2);
		    //plan.f[m][0] = field;// + _Complex_I*0.0;
		    //plan.f[m][1] = 0;// + _Complex_I*0.0;
		    transformer.setY(m, field);
	    	data_out << x1 << " " << x2 << " " << field << endl;
	    	m++;
	    }
    }
    data_out.flush();
    data_out.close();
    transformer.transform();

}


void loadData(const map<string, string>& params) {
	string strDims = Utils::FindProperty(params, "dims", "1");
	vector<string> dimsStr;
	vector<int> dims;
	boost::split(dimsStr, strDims, boost::is_any_of(",;"), boost::token_compress_on);
	for (vector<string>::iterator it = dimsStr.begin() ; it != dimsStr.end(); ++it) {
		if ((*it).length() != 0) {
			dims.push_back(stoi(*it));
			//dims.insert(dims.begin(), stoi(*it));
		}
	}

	assert(dims.size() >= 2);

	int thetaIndex = 1;
	int phiIndex = 2;
	if (dims.size() == 2) {
		thetaIndex = 0;
		phiIndex = 1;
	}
	int thetaDownSample = Utils::FindIntProperty(params, "thetaDownSample", 1);
	int phiDownSample = Utils::FindIntProperty(params, "phiDownSample", 1);

	int numTheta = dims[thetaIndex];
	int numPhi = dims[phiIndex];

	assert(thetaDownSample > 0 && (numTheta % thetaDownSample == 0));
	assert(phiDownSample > 0 && (numPhi % phiDownSample == 0));

	numTheta /= thetaDownSample;
	numPhi /= phiDownSample;

	int M = numTheta * numPhi;
	int N = numPhi;

	int lMax = Utils::FindIntProperty(params, "lMax", 0);
	assert(lMax >= 0);
	if (lMax > 0) {
		N = min(lMax, N);
	}

	bool saveData = Utils::FindIntProperty(params, "saveData", 0);
	bool doReconstruction = Utils::FindIntProperty(params, "doReconstruction", 1);
	bool decompOrPower = Utils::FindIntProperty(params, "decompOrPower", 0);
	Transformer transformer(N, M, Utils::FindProperty(params, RESULTS_OUT, RESULTS_TXT),
		doReconstruction ? Utils::FindProperty(params, RECONST_OUT, RECONST_TXT) : "", decompOrPower);
	transformer.init();

	vector<vector<pair<int, int>>> regions;
	regions.push_back(vector<pair<int, int>>());

	double polarGap = Utils::FindDoubleProperty(params, "polarGap", 15);
	polarGap /= 360; // Convert to NFSFT units
	double latCoef = (0.5 - 2 * polarGap) / dims[thetaIndex];
	double longCoef = 1.0 / dims[phiIndex]; // Currently expanding the wedge to full sphere

	string strType = Utils::FindProperty(params, "type", "snapshot");
	to_lower(strType);
	assert(strType == "snapshot" || strType == "video");
	int type = TYPE_SNAPSHOT;
	if (strType == "video") {
		type = TYPE_VIDEO;
	}

	path dataOut(Utils::FindProperty(params, DATA_OUT, DATA_TXT));
	string filePath = Utils::FindProperty(params, "filePath", "");
	assert(filePath.size() > 0);

	vector<int> varIndices;

    /* define nodes and data*/
	if (type == TYPE_SNAPSHOT) {
		assert(dims.size() == 3);
		const int rIndex = 0;
		int numGhost = Utils::FindIntProperty(params, "numGhost", 3);
		string strNumProcs = Utils::FindProperty(params, "numProcs", "1");
		vector<string> numProcsStr;
		vector<int> numProcs;
		boost::split(numProcsStr, strNumProcs, boost::is_any_of(",;"), boost::token_compress_on);
		for (vector<string>::iterator it = numProcsStr.begin() ; it != numProcsStr.end(); ++it) {
			if ((*it).length() != 0) {
				numProcs.push_back(stoi(*it));
				//numProcs.insert(numProcs.begin(), stoi(*it));
			}
		}

		vector<int> dimsPerProc;

		for (size_t i = 0; i < dims.size(); i++) {
			assert(dims[i] % numProcs[i] == 0);
			int procDim = dims[i] / numProcs[i] + 2 * numGhost;
			//cout << "dimsPerProc[" << i << "]=" << procDim << endl;
			dimsPerProc.push_back(procDim);
		}

		assert(numProcs.size() == dims.size());
		auto numProc = accumulate(numProcs.begin(), numProcs.end(), 1, multiplies<int>());
		//cout << "numProc:" << numProc << endl;

		int layer = Utils::FindIntProperty(params, "layer", 100);
		int timeMoment = Utils::FindIntProperty(params, "timeMoment", -1);
		assert(timeMoment >= 0);

		//double wedgeAngle = Utils::FindDoubleProperty(params, "wedgeAngle", 90);
		int totalNumVars = Utils::FindIntProperty(params, "numVars", 1);

		int varIndex = Utils::FindIntProperty(params, "varIndex", 0);
		varIndices.push_back(varIndex);

		int m = 0;
		vector<vector<double>> data;
		for (int procId = 0; procId < numProc; procId++) {
			vector<int> procMinCoords;
			vector<int> procPositions;
			int procSize = 1;
			for (size_t i = 0; i < dims.size(); i++) {
			//for (int i = dims.size() - 1; i >= 0; i--) {
				int procPos = (procId / procSize) % numProcs[i];
				procPositions.push_back(procPos);
				procSize *= numProcs[i];
				int procMinCoord = procPos * (dimsPerProc[i] - 2 * numGhost);
				//procMinCoords.insert(procMinCoords.begin(), procMinCoord);
				procMinCoords.push_back(procMinCoord);
				//cout << "PROC" << procId << " minCoord for " << i << ": " << procMinCoord << "\n";
			}

			if (filePath[filePath.length() - 1] != '/') {
				filePath += "/";
			}
			string dataFile = filePath + "proc" + to_string(procId) + "/VAR" + to_string(timeMoment);
			cout << "Reading: " << dataFile << endl;
			assert(exists(dataFile));
			BinaryDataLoader dl(dataFile, 1000000, dimsPerProc, regions, totalNumVars, varIndices, TYPE_SNAPSHOT);
			//cout << "procId:" << procId << endl;
			dl.Next();
			assert(dl.GetPageSize() == 1);
			//real time = dl.GetX(0);
			auto y = dl.GetY(0);
			for (int i = 0; i < dl.GetDim(); i++) {
				auto i1 = i;
				vector<int> coords(3);
				//cout << "coords= ";
				for (int j = 0; j < (int) dl.GetDims().size(); j++) {
					int coord = i1 % dl.GetDims()[j];
					//cout << coord << ", ";
					coords[j] = coord;
					//if (j == thetaIndex && (procPositions[j] % 2) != 0) {
					//	//cout << procPositions[j] << endl;
					//	coords[j] = dl.GetDims()[j] - coords[j] - 1;
					//}
					i1 -= coord;
					i1 /= dl.GetDims()[j];
				}
				bool ghost = false;
				for (int j = 0; j < (int) dimsPerProc.size(); j++) {
					if (coords[j] < numGhost || (coords[j] + numGhost) >= dimsPerProc[j]) {
						ghost = true;
						break;
					}
				}
				//cout << ghost << endl;
				if (!ghost) {
					if (coords[rIndex] - numGhost + procMinCoords[rIndex] == layer) {
						int phiCoord = coords[phiIndex] - numGhost + procMinCoords[phiIndex];
						int thetaCoord = coords[thetaIndex] - numGhost + procMinCoords[thetaIndex];
						if ((phiCoord % phiDownSample == 0) && (thetaCoord % thetaDownSample == 0)) {
							double x1 = -0.5 + longCoef * phiCoord;
							double x2 = polarGap + latCoef * thetaCoord;
							if (x1 < -0.5 || x1 > 0.5) {
								cout << "x1 out of range: " << x1 << endl;
								assert(x1 >= -0.5 && x1 <= 0.5);
							}
							if (x2 < 0 || x2 > 0.5) {
								cout << "x2 out of range: " << x2 << endl;
								assert(x2 >= 0 && x2 <= 0.5);
							}
							//cout << "Coords: " << x1 << " " << x2 << "\n";
							int k;
							auto dataIter = data.begin();
							for (k = 0; k < (int) data.size(); k++) {
								if (data[k][0] > x1 || (data[k][0] == x1 && data[k][1] > x2)) {
									break;
								}
								dataIter++;
							}
							data.insert(dataIter, {x1, x2, y[i]});
						}
					}
				}
			}
		}
		ofstream data_out(dataOut.string());
		for (auto& elem : data) {
			assert(m < M);
			data_out << elem[0] << " " << elem[1] << " " << elem[2] << "\n";
			transformer.setX(m, elem[0], elem[1]);
			//plan.x[2*m] = elem[0];
			//plan.x[2*m+1] = elem[1];
			transformer.setY(m, elem[2]);
			//plan.f[m][0] = elem[2];
			//plan.f[m][1] = 0;
			m++;
		}
		assert(m == M);
		data_out.flush();
		data_out.close();
		/* precomputation (for NFFT, node-dependent) */
		cout << "Transforming...";
		cout.flush();
		transformer.transform();
	    cout << "done." << endl;
	} else { // TYPE_VIDEO
		assert(dims.size() == 2);
		int bufferSize = Utils::FindIntProperty(params, "bufferSize", 100000);
		varIndices.push_back(0);
		int startTime = Utils::FindIntProperty(params, "startTime", 0);
		int endTime = Utils::FindIntProperty(params, "endTime", -1);

		string dataFile = filePath;
		cout << "Reading: " << dataFile << endl;
		assert(exists(dataFile));
		BinaryDataLoader dl(dataFile, bufferSize, dims, regions, 1 /*totalNumVars*/, varIndices, TYPE_VIDEO);
		int timeOffset = 0;

		#ifdef BOOST_FILESYSTEM_VER2
			string dataOutDir = dataOut.parent_path().directory_string();
			string dataOutStem = dataOut.stem();
			string dataOutExt = dataOut.extension();
		#else
			string dataOutDir = dataOut.parent_path().string();
			string dataOutStem = dataOut.stem().string();
			string dataOutExt = dataOut.extension().string();
		#endif
		if (!dataOutDir.empty()) {
			dataOutDir += "/";
		}
		while (dl.Next()) {
			//cout << "procId:" << procId << endl;
			for (int t = 0; t < dl.GetPageSize(); t++) {
				if (t + timeOffset < startTime) {
					continue;
				}
				if (endTime >= 0 && t + timeOffset > endTime) {
					break;
				}
				const auto timeIndex = t + timeOffset;
				cout.flush();
				int m = 0;
				real time = dl.GetX(t);
				cout << "Reading time moment " << timeIndex << "(" << time << ")...";
				auto y = dl.GetY(t);
				ofstream* data_out = nullptr;
				if (saveData) {
					data_out = new ofstream(dataOutDir + dataOutStem + to_string(t) + dataOutExt);
				}
				for (int i = 0; i < dl.GetDim(); i++) {
					auto i1 = i;
					vector<int> coords(dl.GetDims().size());
					//cout << "coords= ";
					for (int j = 0; j < (int) dl.GetDims().size(); j++) {
						int coord = i1 % dl.GetDims()[j];
						//cout << coord << ", ";
						coords[j] = coord;
						//if (j == thetaIndex && (procPositions[j] % 2) != 0) {
						//	//cout << procPositions[j] << endl;
						//	coords[j] = dl.GetDims()[j] - coords[j] - 1;
						//}
						i1 -= coord;
						i1 /= dl.GetDims()[j];
					}
					int phiCoord = coords[phiIndex];
					int thetaCoord = coords[thetaIndex];
					if ((phiCoord % phiDownSample == 0) && (thetaCoord % thetaDownSample == 0)) {
						double x1 = -0.5 + longCoef * phiCoord;
						double x2 = polarGap + latCoef * thetaCoord;
						if (x1 < -0.5 || x1 > 0.5) {
							cout << "x1 out of range: " << x1 << endl;
							assert(x1 >= -0.5 && x1 <= 0.5);
						}
						if (x2 < 0 || x2 > 0.5) {
							cout << "x2 out of range: " << x2 << endl;
							assert(x2 >= 0 && x2 <= 0.5);
						}
						//cout << "Coords: " << x1 << " " << x2 << "\n";
						assert(m < M);
						if (data_out) {
							*data_out << x1 << " " << x2 << " " << y[i] << endl;
						}
						if (t == startTime) {
							transformer.setX(m, x1, x2);
							//plan.x[2*m] = x1;
							//plan.x[2*m+1] = x2;
						} else {
							//assert(plan.x[2*m] == x1);
							//assert(plan.x[2*m+1] == x2);
						}
						transformer.setY(m, y[i]);
						//plan.f[m][0] = y[i];
						//plan.f[m][1] = 0;
						m++;
					}
				}
				assert(m == M);
				if (data_out) {
					data_out->flush();
					data_out->close();
					delete data_out;
				}
			    cout << "done." << endl;
				cout << "Transformation for time moment " << timeIndex << "...";
				cout.flush();
				transformer.transform(timeIndex);
			    cout << "done." << endl;
			}
			timeOffset += dl.GetPageSize();
			if (endTime >= 0 && timeOffset > endTime) {
				break;
			}
		}
		transformer.finalize();
	}
}

int main(int argc, char *argv[]) {
	if (argc == 2 && string("-h") == argv[1]) {
		cout << "Usage: ./D2 [param file] [params to overwrite]\nparam file defaults to " << "parameters.txt" << endl;
		return EXIT_SUCCESS;
	}
	string paramFileName = argc > 1 ? argv[1] : "parameters.txt";
	string cmdLineParams = argc > 2 ? argv[2] : "";
	boost::replace_all(cmdLineParams, " ", "\n");

	if (!exists(paramFileName)) {
		cout << "Cannot find " << paramFileName << endl;
		return EXIT_FAILURE;
	}

	map<string, string> paramsFromFile = Utils::ReadProperties(paramFileName);
	map<string, string> params = Utils::ReadPropertiesFromString(cmdLineParams);
	for (const auto& entry : paramsFromFile) {
		params.insert({entry.first, entry.second});
	}

    #ifdef TEST
    	loadTestData(params);
    #else
    	loadData(params);
    #endif
    return EXIT_SUCCESS;
}
