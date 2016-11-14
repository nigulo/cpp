#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>

//#include <complex.h>
#include <omp.h>
#include "nfft3.h" /* NFFT3 header */

#include <fstream>
#include <vector>
#include "BinaryDataLoader.h"

#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include "utils/utils.h"
#include <iomanip>

//#define TEST
//#define ONLY_AZIMUTHAL_RECONST

using namespace utils;
using namespace std;
using namespace boost;
using namespace boost::filesystem;

string paramFileName;
nfsft_plan plan; /* transform plan */

void init(int N /* bandwidth/maximum degree */, int M /* number of nodes */) {
	//printf("num_threads = %d\n", omp_get_max_threads());
    printf("nthreads = %d\n", nfft_get_num_threads());
    /* init */
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    /* precomputation (for fast polynomial transform) */
    nfsft_precompute(N,1000.0,0U,0U);

    /* Initialize transform plan using the guru interface. All input and output
     * arrays are allocated by nfsft_init_guru(). Computations are performed with
     * respect to L^2-normalized spherical harmonics Y_k^n. The array of spherical
     * Fourier coefficients is preserved during transformations. The NFFT uses a
     * cut-off parameter m = 6. See the NFFT 3 manual for details.
     */
    nfsft_init(&plan, N, M);
    //nfsft_init_guru(&plan, N, M, NFSFT_MALLOC_X | NFSFT_MALLOC_F |
    //    NFSFT_MALLOC_F_HAT | NFSFT_NORMALIZED | NFSFT_PRESERVE_F_HAT,
    //    PRE_PHI_HUT | PRE_PSI | FFTW_INIT | FFT_OUT_OF_PLACE, 6);
}

void finalize() {
	/* Finalize the plan. */
	nfsft_finalize(&plan);

	/* Destroy data precomputed for fast polynomial transform. */
	nfsft_forget();

}

void transform(int suffix = -1) {
    /* precomputation (for NFFT, node-dependent) */
	if (suffix <= 0) { // first time
		nfsft_precompute_x(&plan);
	}

    /* Direct adjoint transformation, display result. */
    nfsft_adjoint_direct(&plan);

    string suffixStr;
    if (suffix >= 0) {
    	suffixStr = to_string(suffix);
    }

    ofstream decomp_out(string("decomp") + suffixStr + ".txt");
    for (int k = 0; k <= plan.N; k++) {
        for (int n = -k; n <= k; n++) {
        	decomp_out << k << " " << n << " " << plan.f_hat[NFSFT_INDEX(k,n,&plan)][0] << " " << plan.f_hat[NFSFT_INDEX(k,n,&plan)][1] << "\n";
        }
    }
    decomp_out.flush();
    decomp_out.close();

	#ifdef ONLY_AZIMUTHAL_RECONST
    	// Set all nonazimuthal waves to zero
		for (int k = 0; k <= plan.N; k++) {
			for (int n = -k+1; n <= k-1; n++) {
				plan.f_hat[NFSFT_INDEX(k,n,&plan)][0] = 0;
				plan.f_hat[NFSFT_INDEX(k,n,&plan)][1] = 0;
			}
		}
	#endif
    /* Direct transformation, display result. */
    nfsft_trafo_direct(&plan);

    ofstream reconst_out(string("reconst") + suffixStr + ".txt");
    for (int m = 0; m < plan.M_total; m++) {
    	reconst_out << plan.x[2*m] << " " << plan.x[2*m + 1] << " " << plan.f[m][0] << " " << plan.f[m][1] << "\n";
    }
    reconst_out.flush();
    reconst_out.close();

    if (suffix < 0) { // only single transform
    	finalize();
    }
}

void loadTestData() {
	cout << "Generating test data!" << endl;
    ofstream data_out("data.txt");
	const int m_phi = 75;
	const int m_theta = 200;
	int N = m_phi;
	int M = m_phi * m_theta;
    init(N, M);
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
		    plan.x[2*m] = x1;
		    plan.x[2*m+1] = x2;
		    double field = /*sin(2*M_PI*(x1+0.5)) +*/ sin(2*2*M_PI*(x1+0.5)) * sin(3*4*M_PI*x2);
		    plan.f[m][0] = field;// + _Complex_I*0.0;
		    plan.f[m][1] = 0;// + _Complex_I*0.0;
	    	data_out << x1 << " " << x2 << " " << field << endl;
	    	m++;
	    }
    }
    data_out.flush();
    data_out.close();
    transform();

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

	init(N, M);

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

	string filePath = Utils::FindProperty(params, string("filePath"), "");
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
		ofstream data_out("data.txt");
		for (auto& elem : data) {
			assert(m < M);
			data_out << elem[0] << " " << elem[1] << " " << elem[2] << "\n";
			plan.x[2*m] = elem[0];
			plan.x[2*m+1] = elem[1];
			plan.f[m][0] = elem[2];
			plan.f[m][1] = 0;
			m++;
		}
		assert(m == M);
		data_out.flush();
		data_out.close();
		/* precomputation (for NFFT, node-dependent) */
		cout << "Transforming...";
		cout.flush();
		transform();
	    cout << "done." << endl;
	} else { // TYPE_VIDEO
		assert(dims.size() == 2);
		int bufferSize = Utils::FindIntProperty(params, "bufferSize", 100000);
		varIndices.push_back(0);

		string dataFile = filePath;
		cout << "Reading: " << dataFile << endl;
		assert(exists(dataFile));
		BinaryDataLoader dl(dataFile, bufferSize, dims, regions, 1 /*totalNumVars*/, varIndices, TYPE_VIDEO);
		int timeOffset = 0;
		while (dl.Next()) {
			//cout << "procId:" << procId << endl;
			for (int t = 0; t < dl.GetPageSize(); t++) {
				const auto timeIndex = t + timeOffset;
				cout.flush();
				int m = 0;
				real time = dl.GetX(t);
				cout << "Reading time moment " << timeIndex << "(" << time << ") ...";
				auto y = dl.GetY(t);
				ofstream data_out("data" + to_string(t) + ".txt");
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
						data_out << x1 << " " << x2 << " " << y[i] << endl;
						if (t == 0) {
							plan.x[2*m] = x1;
							plan.x[2*m+1] = x2;
						} else {
							assert(plan.x[2*m] == x1);
							assert(plan.x[2*m+1] == x2);
						}
						plan.f[m][0] = y[i];
						plan.f[m][1] = 0;
						m++;
					}
				}
				assert(m == M);
				data_out.flush();
				data_out.close();
			    cout << "done." << endl;
				cout << "Transformation for time moment " << timeIndex << "...";
				cout.flush();
				transform(timeIndex);
			    cout << "done." << endl;
			}
			timeOffset += dl.GetPageSize();
		}
		finalize();
	}
}

int main(int argc, char *argv[]) {

    #ifdef TEST
    	loadTestData();
    #else
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
    	loadData(params);
    #endif
    return EXIT_SUCCESS;
}
