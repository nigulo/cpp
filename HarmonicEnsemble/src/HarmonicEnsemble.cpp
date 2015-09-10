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


int main(int argc, char *argv[]) {
	if (argc == 2 && string("-h") == argv[1]) {
		cout << "Usage: ./HarmonicEnsemble [paramfile]\nparamfile defaults to " << PARAMETERS_FILE << endl;
		return EXIT_SUCCESS;
	}

	string paramFileName = argc > 1 ? argv[1] : PARAMETERS_FILE;

	if (!exists(PARAMETERS_FILE)) {
		cout << "Cannot find " << PARAMETERS_FILE << endl;
		return EXIT_FAILURE;
	}

	map<string, string> params = Utils::ReadProperties(paramFileName);

	double freqMean = Utils::FindIntProperty(params, "freqMean", 1);
	double freqStdDev = Utils::FindIntProperty(params, "freqStdDev", 0.1);
	double ampMean = Utils::FindIntProperty(params, "ampMean", 1);
	double ampStdDev = Utils::FindIntProperty(params, "ampStdDev", 0.1);
	double durationMean = Utils::FindIntProperty(params, "durationMean", 0); // 0 is infinite
	double durationStdDev = Utils::FindIntProperty(params, "durationStdDev", 0.1);
	size_t ensembleSize = Utils::FindIntProperty(params, "ensembleSize", 10);
	double timeSpan = Utils::FindDoubleProperty(params, "timeSpan", 1000);
	double timeStep = Utils::FindDoubleProperty(params, "timeStep", 1);

	HarmonicEnsemble he(ensembleSize,
			freqMean, freqStdDev,
			ampMean, ampStdDev,
			durationMean,
			durationStdDev,
			timeStep);

	size_t count = timeSpan / timeStep;

	ofstream output("output.txt");
	for (size_t i = 0; i < count; i++) {
		double val = he.NextStep();
		output << timeStep * i  << " " << val << endl;

	}
	output.close();

	return EXIT_SUCCESS;
}

double HarmonicEnsemble::NextStep() {
	time = timeStep * no++;
	for (auto i = ensemble.begin(); i != ensemble.end(); i++) {
		if (i->second < time) {
			ensemble.erase(i);
		}
	}
	auto currentSize = ensemble.size();
	for (size_t i = 0; i < size - currentSize; i++) {
		double freq = freqDist(gen);
		double amp = ampDist(gen);
		double duration = durationDist(gen);
		double phase = phaseDist(gen);
		ensemble.push_back(pair<Harmonic, double>(Harmonic(freq, amp, phase), time + duration));
	}
	return 0;
}


