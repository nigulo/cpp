#include <cstring>
#include <cctype>
#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <tuple>
#include <boost/algorithm/string.hpp>
#include "boost/filesystem.hpp"
#include "spline.h"

using namespace std;
using namespace boost::filesystem;

string fileName;
string prefix;
double precision = 0.01;
bool header = true;
double xStep = 1;
vector<double> xs;
double xRange;

/*
pair<double, double> meanVariance(const vector<double>& dat) {
	assert(dat.size() > 0);
	double sum = 0;
	double sumSquares = 0;
	for(auto val = dat.begin(); val != dat.end(); val++) {
		sum += (*val);
		sumSquares += (*val) * (*val);
	}
	int n = dat.size();
	double mean = sum / n;
	return {mean,  sumSquares / n - mean * mean};
}
*/
void multComplex(double x[2], double y[2]) {
	double x0 = x[0];
	x[0] = x0 * y[0] - x[1] * y[1];
	x[1] = x[1] * y[0] + x0 * y[1];
}

void shiftPhase(int n, fftw_complex* data, double phaseShift) {
	int nDiv2 = n >> 1;
	double c1[2] = {cos(-phaseShift), sin(-phaseShift)};
	double c2[2] = {cos(phaseShift), sin(phaseShift)};
	if (phaseShift != 0) {
		for (int i = 1; i < n; i++) {
			if (i != nDiv2) {
				if (i < nDiv2) {
					multComplex(data[i], c1);
				} else if (i > nDiv2) {
					multComplex(data[i], c2);
				}
			}
		}
	}
}

void fft(bool direction, int n, fftw_complex* data, double phaseShift) {
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    memcpy(in, data, sizeof(fftw_complex) * n);
	shiftPhase(n, in, phaseShift);
	fftw_plan p = fftw_plan_dft_1d(n, in, out, direction ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */
    memcpy(data, out, sizeof(fftw_complex) * n);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

}

double getTangent(int i, const vector<double>& data) {
	return (data[i + 1] - data[i - 1]) / 2;
}

double getRealTangent(int i, fftw_complex* data) {
	return (data[i + 1][0] - data[i - 1][0]) / 2;
}

double /*meanEnergy*/ analyticSignal(const vector<double>& realSignal, int modeNo) {
	unsigned n = realSignal.size();
    fftw_complex* conjugatedSignal = (fftw_complex*) malloc(sizeof(fftw_complex) * n);
    for (unsigned i = 0; i < n; i++) {
    	conjugatedSignal[i][0] = realSignal[i];
    	conjugatedSignal[i][1] = 0;
    }
	fft(true, realSignal.size(), conjugatedSignal, 0);
	fft(false, realSignal.size(), conjugatedSignal, M_PI_2);
	//double* amplitude = (double*) malloc(sizeof(double) * n);
	//double* phase = (double*) malloc(sizeof(double) * n);
	//double* frequency = (double*) malloc(sizeof(double) * (n - 2));
	ofstream imfStream(prefix + "_imf_" + to_string(modeNo) + ".csv");
	ofstream ampStream(prefix + "_amp_" + to_string(modeNo) + ".csv");
	ofstream freqStream(prefix + "_freq_" + to_string(modeNo) + ".csv");
	double totalEnergy = 0;
	for (unsigned i = 0; i < n; i++) {
		double x = xs[i];
		double u = realSignal[i];
		double v = conjugatedSignal[i][0] / n; // the fft is unnormalized
		double u2v2 = u * u + v * v;
		double amplitude = sqrt(u2v2);
		totalEnergy += u2v2;
		//double phase = atan(v / u);
		imfStream << x << " " << u << endl;
		ampStream << x << " " << amplitude << endl;
		if (i > 0 && i < n - 1) {
			double frequency = (getRealTangent(i, conjugatedSignal) * u / n - getTangent(i, realSignal) * v) / u2v2 / xStep;
			freqStream << x << " " << frequency << endl;
		}
	}
	imfStream.close();
	ampStream.close();
	freqStream.close();
	return totalEnergy / n;
}

pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>> findExtrema(const vector<double>& xs, const vector<double>& ys) {
	vector<double>* minima_x = new vector<double>();
	vector<double>* minima_y = new vector<double>();
	vector<double>* maxima_x = new vector<double>();
	vector<double>* maxima_y = new vector<double>();
	int j = 0;
	for (unsigned i = 1; i < ys.size() -  1; i++) {
		double y1 = ys[i];
		double y2 = ys[i + 1];
		if (y1 != y2) {
			double y0 = ys[j]; // in a regular case j = i - 1
			if (y1 - y0 < 0 && y1 - y2 < 0 ){
				minima_y->push_back(y1);
				minima_x->push_back(xs[i]);
			} else if (y1 - y0 > 0 && y1 - y2 > 0 ) {
				maxima_y->push_back(y1);
				maxima_x->push_back(xs[i]);
			}
			j = i;
		}
	}
	return {{minima_x, minima_y}, {maxima_x, maxima_y}};
}
/*
const vector<double>* findZeroCrossings(const vector<double>& xs, const vector<double>& ys) {
	vector<double>* zeroCrossings = new vector<double>();
	for (int i = 0; i < ys.size() - 1; i++) {
		double y1 = ys[i];
        if (y1 == 0) {
        	zeroCrossings->push_back(xs[i]);
        } else {
    		double y2 = ys[i + 1];
        	if ((y1 > 0 && y2 <= 0) || (y1 < 0 && y2 >= 0)) {
            	zeroCrossings->push_back(abs(y1) < abs(y2) ? xs[i] : xs[i + 1]);
        	}
        }
	}
	return zeroCrossings;
}
*/

int findNumZeroCrossings(const vector<double>& xs, const vector<double>& ys) {
	int numZeroCrossings = 0;
	for (unsigned i = 0; i < ys.size() - 1; i++) {
		double y1 = ys[i];
        if (y1 == 0) {
        	numZeroCrossings++;
        } else {
    		double y2 = ys[i + 1];
        	if ((y1 > 0 && y2 <= 0) || (y1 < 0 && y2 >= 0)) {
            	numZeroCrossings++;
        	}
        }
	}
	return numZeroCrossings;
}

pair<int, pair<double, double>> imfStep(vector<double>& imf, pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>>& stepExtrema) {
	tk::spline lowerEnv;
	lowerEnv.set_points(*stepExtrema.first.first, *stepExtrema.first.second);
	tk::spline upperEnv;
	upperEnv.set_points(*stepExtrema.second.first, *stepExtrema.second.second);
	//vector<double>* dat1 = new vector<double>(dat.size());
	for (unsigned i = 0; i < imf.size(); i++) {
		double x = xs[i];
		imf[i] = imf[i] - (lowerEnv(x) + upperEnv(x)) / 2;
	}
	delete stepExtrema.first.first;
	delete stepExtrema.first.second;
	delete stepExtrema.second.first;
	delete stepExtrema.second.second;
	auto newExtrema = findExtrema(xs, imf);
	int numExtrema = newExtrema.first.first->size() + newExtrema.second.first->size();
	int numZeroCrossings = findNumZeroCrossings(xs, imf);
    //cout << endl << "NE: " << numExtrema << ", NZC: " << numZeroCrossings;
	if (abs(numExtrema - numZeroCrossings) <= 1) {
		double extremaStart = max(*(newExtrema.first.first->begin()), *(newExtrema.second.first->begin()));
		double extremaEnd = min(*(newExtrema.first.first->end() - 1), *(newExtrema.second.first->end() - 1));
		delete newExtrema.first.first;
		delete newExtrema.first.second;
		delete newExtrema.second.first;
		delete newExtrema.second.second;
		return {numZeroCrossings, {extremaStart, extremaEnd}};
	}
	return imfStep(imf, newExtrema);
}

pair<const vector<double>* /*imf*/, double /*avgFreq*/> imf(int modeNo, vector<double>& dat) {
	assert(xs.size() == dat.size());
	auto extrema = findExtrema(xs, dat);
	int numExtrema = min(extrema.first.first->size(), extrema.second.first->size());
	if (numExtrema <= 3) {
		delete extrema.first.first;
		delete extrema.first.second;
		delete extrema.second.first;
		delete extrema.second.second;
		return {&dat, 0};
	}
	cout << "Extracting mode " << modeNo << " (" << numExtrema << ") ...";
	cout.flush();
	vector<double>* imf = new vector<double>(dat);
	auto imfResult = imfStep(*imf, extrema);
    int numZeroCrossings = imfResult.first;
    auto extremaStart = imfResult.second.first;
    auto extremaEnd = imfResult.second.second;
    cout << " done." << endl;

    auto xsIter = xs.begin();
    auto datIter = dat.begin();
    auto imfIter = imf->begin();
    for (;xsIter != xs.end();) {
    	double x = *xsIter;
    	if (x < extremaStart || x > extremaEnd) {
    		xsIter = xs.erase(xsIter);
    		datIter = dat.erase(datIter);
    		imfIter = imf->erase(imfIter);
    	} else {
    		*datIter -= *imfIter;
    		xsIter++;
    		datIter++;
    		imfIter++;
    	}
    }
	xRange = *(xs.end() - 1) - *(xs.begin());

    //for (unsigned i = 0; i < dat.size(); i++) {
   	//	dat[i] -= (*imf)[i];
    //}
    return {imf, 0.5 * numZeroCrossings / xRange};
}

pair<double, double> getLatR(const string& fileName) {
	int index = fileName.find('_');
	string latStr = fileName.substr(0, index);
	string rStr = fileName.substr(index + 1);
	double lat = -1;
	for (unsigned i = 0; i < latStr.length(); i++) {
		if (isdigit(latStr[i])) {
			lat = stod(latStr.substr(i));
			break;
		}
	}
	double r = -1;
	for (unsigned i = 0; i < rStr.length(); i++) {
		if (!isdigit(rStr[i])) {
			r = stod(rStr.substr(0, i));
			break;
		}
	}
	return {lat, r};
}

void collect() {
	vector<vector<tuple<double, double, double, double>>> allModes;
	directory_iterator end_itr; // default construction yields past-the-end
	path currentDir(".");
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (fileName.substr(fileName.length() - 4) != ".log") {
				continue;
			}
		    cout << "Processing " << fileName << endl;
			auto latR = getLatR(fileName);
			double lat = latR.first;
			double r = latR.second;
			vector<double[4]> modes;
			ifstream input(fileName);
			unsigned modeNo = 0;
			for (string line; getline(input, line);) {
				//cout << line << endl;
				std::vector<std::string> words;
				boost::split(words, line, boost::is_any_of("\t "), boost::token_compress_on);
				for (vector<string>::iterator it = words.begin() ; it != words.end(); ++it) {
					if ((*it).length() == 0) {
						words.erase(it);
					}
				}
				//double mode[4] = {latR.first, latR.second, stod(words[1]), stod(words[2])};
				if (modeNo >= allModes.size()) {
					allModes.push_back(vector<tuple<double, double, double, double>>());
				}
				vector<tuple<double, double, double, double>>& mode = allModes[modeNo];
				auto i = mode.begin();
				for (; i != mode.end(); i++) {
					double currentLat = get<0>(*i);
					double currentR = get<1>(*i);
					if (lat < currentLat || (lat == currentLat && r < currentR)) {
						break;
					}
				}
				mode.insert(i, make_tuple(lat, r, stod(words[1]), stod(words[2])));
				modeNo++;
		    }
			input.close();
		}
	}
	for (unsigned i = 0; i < allModes.size(); i++) {
		string modeNo = to_string(i + 1);
		ofstream enStream(string("ens") + modeNo + ".csv");
		ofstream freqStream(string("freqs") + modeNo + ".csv");
		for (unsigned j = 0; j < allModes[i].size(); j++) {
			auto dat = allModes[i][j];
			double lat = get<0>(dat);
			double r = get<1>(dat);
			double freq = get<2>(dat);
			double en = get<3>(dat);
			if (j > 0 && lat != get<0>(allModes[i][j - 1])) {
				enStream << endl;
				freqStream << endl;
			}
			enStream << lat << " " << r << " " << en << endl;
			freqStream << lat << " " << r << " " << freq << endl;
		}
		enStream.close();
		freqStream.close();
	}
}

int main(int argc, char** argv) {
	if (argc == 1) {
		collect();
		return EXIT_SUCCESS;
	}
	fileName = argv[1];
    cout << "Processing " << fileName << endl;
	string::size_type n = fileName.find('.');
	prefix = fileName.substr(0, n);

	// Check if IMF-s already calculated
	if (exists(prefix + ".log")) {
		return EXIT_SUCCESS;
	}

	vector<double> ys;
	ifstream input(fileName);
	for (string line; getline(input, line);) {
		//cout << line << endl;
		std::vector<std::string> words;
		boost::split(words, line, boost::is_any_of("\t "), boost::token_compress_on);
		for (vector<string>::iterator it = words.begin() ; it != words.end(); ++it) {
			if ((*it).length() == 0) {
				words.erase(it);
			}
		}
		if (words.size() > 0 && words[0][0] == '#') {
			//cout << "Skipping comment line: " << line << endl;
		} else if (words.size() == 2) {
			try {
				double xVal = stod(words[0]);
				double yVal = stod(words[1]);
				xs.push_back(xVal);
				ys.push_back(yVal);
			} catch (invalid_argument& ex) {
				cout << "Skipping line, invalid number: " << line << endl;
			}
		} else {
			cout << "Skipping line, invalid number of columns: " << line << endl;
		}
    }
	input.close();
	if (xs.size() > 1) {
		xStep = xs[1] - xs[0]; // Assuming even sampling
		xRange = *(xs.end() - 1) - *(xs.begin());
		n = xs.size();
		//cout << "xStep: " << xStep << ", xRange: " << xRange << endl;
	}

	stringstream logText;
	for (int modeNo = 1;; modeNo++) {
		cout << "Before IMF " << modeNo << " " << ys.size() << endl;
		auto imfAndFreq = imf(modeNo, ys);
		cout << "After IMF "<< endl;
		if (imfAndFreq.second == 0) {
			break;
		}
		cout << "Before AS"<< endl;
		double meanEnergy = analyticSignal(*imfAndFreq.first, modeNo);
		cout << "After AS1 "<< endl;
        logText << modeNo << ": " << imfAndFreq.second << " " << meanEnergy << endl;
		cout << "After AS2 "<< endl;
	}
	ofstream logStream(prefix + ".log");
	logStream << logText.str();
	logStream.close();
	cout << logText.str();
	return EXIT_SUCCESS;
}
