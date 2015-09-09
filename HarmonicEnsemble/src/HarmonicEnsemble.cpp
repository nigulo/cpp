//============================================================================
// Name        : HarmonicEnsemble.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
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

using namespace boost;
using namespace boost::filesystem;
using namespace utils;

#define PARAMETERS_FILE "parameters.txt"

/*
class Harmonic {
	double freq;
	double phase;
	double startTime;
	double duration;
};
*/

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
	double freqVar = Utils::FindIntProperty(params, "freqVar", 0.1);
	double durationMean = Utils::FindIntProperty(params, "durationMean", 0); // 0 is infinite
	double durationVar = Utils::FindIntProperty(params, "durationVar", 0.1);
	size_t ensembleSize = Utils::FindIntProperty(params, "ensembleSize", 10);
	double timeSpan = Utils::FindDoubleProperty(params, "timeSpan", 1000);

	normal_distribution<> frecDist(freqMean, freqVar);
	normal_distribution<> durationDist(durationMean, durationVar);
	uniform_real_distribution<> phaseDist(0, 1);

	double time = 0;
	vector<tuple<double /*freq*/, double /*phase*/>>

	std::map<int, int> hist;
	for(int n=0; n<10000; ++n) {
		++hist[std::round(d(gen))];
	}
	for(auto p : hist) {
		std::cout << std::fixed << std::setprecision(1) << std::setw(2)
				  << p.first << ' ' << std::string(p.second/200, '*') << '\n';
	}

	return EXIT_SUCCESS;
}
