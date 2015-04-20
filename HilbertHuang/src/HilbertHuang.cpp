#include <cstring>
#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>
//#include <iostream>
#include <fstream>
//#include <cstdlib>
//#include <sstream>
#include <boost/algorithm/string.hpp>

using namespace std;

string fileName;
string prefix;
double precision = 0.01;
bool header = true;
double xScale = 1;

void multComplex(double x[2], double y[2]) {
	double x0 = x[0];
	x[0] = x0 * y[0] - x[1] * y[1];
	x[1] = x[1] * y[0] + x0 * y[1];
}

void shift(int n, fftw_complex* data, double phaseShift) {
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
	shift(n, in, phaseShift);
	fftw_plan p = fftw_plan_dft_1d(n, in, out, direction ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */
    memcpy(data, out, sizeof(fftw_complex) * n);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

}

double getTangent(int i, double* data) {
	return (data[i + 1] - data[i - 1]) / 2;
}

double getRealTangent(int i, fftw_complex* data) {
	return (data[i + 1][0] - data[i - 1][0]) / 2;
}

void analyticSignal(int n, double* realSignal) {
    fftw_complex* conjugatedSignal = (fftw_complex*) malloc(sizeof(fftw_complex) * n);
    for (int i = 0; i < n; i++) {
    	conjugatedSignal[i][0] = realSignal[i];
    	conjugatedSignal[i][1] = 0;
    }
	fft(true, n, conjugatedSignal, 0);
	fft(false, n, conjugatedSignal, (M_PI_2));
	double* amplitude = (double*) malloc(sizeof(double) * n);
	double* phase = (double*) malloc(sizeof(double) * n);
	double* frequency = (double*) malloc(sizeof(double) * (n - 2));
	for (int i = 0; i < n; i++) {
		double u = realSignal[i];
		double v = conjugatedSignal[i][0];
		double u2v2 = u * u + v * v;
		amplitude[i] = sqrt(u2v2);
		phase[i] = atan(v / u);
		if (i > 0 && i < n - 1) {
			frequency[i] = (getRealTangent(i, conjugatedSignal) * u - getTangent(i, realSignal) * v) / u2v2;
		}
	}
}

int main(int argc, char** argv) {
	fileName = argv[0];
	string::size_type n = fileName.find('.');
	prefix = fileName.substr(0, n);

	ifstream input(fileName);
	vector<double> xs, ys;
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
		xScale = xs[1] - xs[0]; // Assuming even sampling
	}

	return 0;
}
