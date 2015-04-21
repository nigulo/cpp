#include <cstring>
#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>
//#include <iostream>
#include <fstream>
//#include <cstdlib>
//#include <sstream>
#include <boost/algorithm/string.hpp>
#include "spline.h"

using namespace std;

string fileName;
string prefix;
double precision = 0.01;
bool header = true;
double xStep = 1;
vector<double> xs;
double xRange;

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
    fftw_complex* conjugatedSignal = (fftw_complex*) malloc(sizeof(fftw_complex) * realSignal.size());
    for (unsigned i = 0; i < realSignal.size(); i++) {
    	conjugatedSignal[i][0] = realSignal[i];
    	conjugatedSignal[i][1] = 0;
    }
	fft(true, realSignal.size(), conjugatedSignal, 0);
	fft(false, realSignal.size(), conjugatedSignal, (M_PI_2));
	//double* amplitude = (double*) malloc(sizeof(double) * n);
	//double* phase = (double*) malloc(sizeof(double) * n);
	//double* frequency = (double*) malloc(sizeof(double) * (n - 2));
	ofstream imfStream(prefix + "_imf_" + to_string(modeNo) + ".csv");
	ofstream ampStream(prefix + "_amp_" + to_string(modeNo) + ".csv");
	ofstream freqStream(prefix + "_freq_" + to_string(modeNo) + ".csv");
	double totalEnergy = 0;
	for (unsigned i = 0; i < realSignal.size(); i++) {
		double x = xs[i];
		double u = realSignal[i];
		double v = conjugatedSignal[i][0];
		double u2v2 = u * u + v * v;
		double amplitude = sqrt(u2v2);
		totalEnergy += u2v2;
		//double phase = atan(v / u);
		imfStream << x << " " << v << endl;
		ampStream << x << " " << amplitude << endl;
		if (i > 0 && i < realSignal.size() - 1) {
			double frequency = (getRealTangent(i, conjugatedSignal) * u - getTangent(i, realSignal) * v) / u2v2 / xStep;
			freqStream << x << " " << frequency << endl;
		}
	}
	imfStream.close();
	ampStream.close();
	freqStream.close();
	return totalEnergy / realSignal.size();
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

int /*numZeroCrossings*/ imfStep(vector<double>& imf, pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>>& stepExtrema) {
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
	auto numExtrema = newExtrema.first.first->size() + newExtrema.second.first->size();
	int numZeroCrossings = findNumZeroCrossings(xs, imf);
    cout << endl << "NE: " << numExtrema << ", NZC: " << numZeroCrossings;
	if (numExtrema - numZeroCrossings <= 1) {
		delete newExtrema.first.first;
		delete newExtrema.first.second;
		delete newExtrema.second.first;
		delete newExtrema.second.second;
		return numZeroCrossings;
	}
	return imfStep(imf, newExtrema);
}

pair<const vector<double>* /*imf*/, double /*avgFreq*/> imf(int modeNo, vector<double>& dat) {
	auto extrema = findExtrema(xs, dat);
	auto numExtrema = min(extrema.first.first->size(), extrema.second.first->size());
	if (numExtrema <= 2) {
		delete extrema.first.first;
		delete extrema.first.second;
		delete extrema.second.first;
		delete extrema.second.second;
		return {&dat, 0};
	}
	cout << "Extracting mode " << modeNo << " (" << numExtrema << ") ... ";
	vector<double>* imf = new vector<double>(dat);
    int numZeroCrossings = imfStep(*imf, extrema);
    cout << " done." << endl;

    for (unsigned i = 0; i < dat.size(); i++) {
        dat[i] -= (*imf)[i];
    }
    return {imf, 0.5 * numZeroCrossings / xRange};
}

void collect() {}

int main(int argc, char** argv) {
	if (argc == 1) {
		collect();
		return EXIT_SUCCESS;
	}
	fileName = argv[1];
	string::size_type n = fileName.find('.');
	prefix = fileName.substr(0, n);

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
			cout << "Skipping comment line: " << line << endl;
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
		xRange = xs.end() - xs.begin();
		n = xs.size();
		cout << "xStep: " << xStep << ", xRange: " << xRange << endl;
	}

	stringstream logText;
	for (int modeNo = 1;; modeNo++) {
		auto imfAndFreq = imf(modeNo, ys);
		if (imfAndFreq.second == 0) {
			break;
		}
		double meanEnergy = analyticSignal(*imfAndFreq.first, modeNo);
        logText << modeNo << ": " << imfAndFreq.second << " " << meanEnergy << endl;
	}
	ofstream logStream(prefix + ".log");
	logStream << logText.str();
	logStream.close();
	cout << logText.str();
	return EXIT_SUCCESS;
}
