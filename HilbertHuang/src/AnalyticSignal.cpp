#include "AnalyticSignal.h"
#include <fftw3.h>
#include <math.h>
#include <cstring>
#include <fstream>

AnalyticSignal::AnalyticSignal() {
}

AnalyticSignal::~AnalyticSignal() {
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

double AnalyticSignal::calculate(const TimeSeries& ts, int modeNo, const string& prefix) {
	const vector<double>& xs = ts.getXs();
	const vector<double>& realSignal = ts.getYs();
	unsigned n = realSignal.size();
	double 	xStep = xs[1] - xs[0]; // Assuming even sampling
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
