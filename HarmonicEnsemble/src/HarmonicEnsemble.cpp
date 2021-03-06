//============================================================================
// Name        : HarmonicEnsemble.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include "utils.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <math.h>
#include <sstream>
#include <memory>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "HarmonicEnsemble.h"

using namespace boost;
using namespace boost::filesystem;
using namespace utils;

#define PARAMETERS_FILE "parameters.txt"

#define SQUARE(x) ((x) * (x))
#define GAUSS(x, mean, stdev) (1 / ((stdev) * sqrt(2 * M_PI)) * exp(-SQUARE(x - mean) / (2 * SQUARE(stdev))))
#define NGAUSS(x, mean, stdev) (exp(-SQUARE(x - mean) / (2 * SQUARE(stdev))))

int main(int argc, char *argv[]) {
	if (argc == 2 && string("-h") == argv[1]) {
		cout << "Usage: ./HarmonicEnsemble [paramfile]\nparamfile defaults to " << PARAMETERS_FILE << endl;
		return EXIT_SUCCESS;
	}

	string paramFileName = argc > 1 ? argv[1] : PARAMETERS_FILE;

	if (!exists(paramFileName)) {
		cout << "Cannot find " << paramFileName << endl;
		return EXIT_FAILURE;
	}

	map<string, string> params = Utils::ReadProperties(paramFileName);

	double freqMean = Utils::FindDoubleProperty(params, "freqMean", 1);
	double freqStdDev = Utils::FindDoubleProperty(params, "freqStdDev", 0.1);
	double ampMax = Utils::FindDoubleProperty(params, "ampMax", 1);
	double ampMaxFreq = Utils::FindDoubleProperty(params, "ampMaxFreq", 0); // equal amplitudes
	double durationMean = Utils::FindDoubleProperty(params, "durationMean", 10);
	double durationStdDev = Utils::FindDoubleProperty(params, "durationStdDev", 1);
	size_t ensembleSize = Utils::FindIntProperty(params, "ensembleSize", 10);
	double timeSpan = Utils::FindDoubleProperty(params, "timeSpan", 1000);
	double timeStepMean = Utils::FindDoubleProperty(params, "timeStep", 0.1);
	double timeStepStdDev = Utils::FindDoubleProperty(params, "timeStepStdDev", 0);

	HarmonicEnsemble he(ensembleSize,
			freqMean, freqStdDev,
			ampMax, ampMaxFreq,
			durationMean,
			durationStdDev,
			timeStepMean,
			timeStepStdDev);

	size_t count = timeSpan / timeStepMean;
	ofstream output("output.txt");
	for (size_t i = 0; i < count; i++) {
		pair<double, double> val = he.NextStep();
		output << val.first  << " " << val.second << endl;

	}
	output.close();

	return EXIT_SUCCESS;
}

pair<double, double> HarmonicEnsemble::NextStep() {
	auto i = ensemble.begin();
	while (i != ensemble.end()) {
		if (i->second < time) {
			ensemble.erase(i);
		} else {
			i++;
		}
	}
	auto currentSize = ensemble.size();
	for (size_t i = 0; i < size - currentSize; i++) {
		double freq = freqDist(gen);
		double amp = ampMax;
		if (ampMaxFreq > 0) {
			double ampStDev1 = (3 * freqStdDev + ampMaxFreq - freqMean) / 3;
			double ampStDev2 = 2 * freqStdDev - ampStDev1;
			if (freq < ampMaxFreq) {
				amp *= NGAUSS(freq, ampMaxFreq, ampStDev1);
			} else if (freq > ampMaxFreq) {
				amp *= NGAUSS(freq, ampMaxFreq, ampStDev2);
			}
		}
		double duration = durationDist(gen);
		if (freq > 0 && amp > 0 && duration > 0) {
			double phase = phaseDist(gen);
			cout << "Adding harmonic: ";
			cout << freq;
			cout << " " << amp;
			cout << " " << phase;
			cout << " " << duration;
			cout << endl;
			ensemble.push_back(pair<Harmonic, double>(Harmonic(freq, amp, phase), time + duration));
		}
	}
	double val = 0;
	for (auto& element : ensemble) {
		val += element.first.GetValue(time);
	}
	pair<double, double> retVal(time, val);
	double dt;
	do {
		dt = timeStepDist(gen);
	} while (dt <= 0);
	time += dt;
	return retVal;
}


