#include "HilbertHuang.h"
#include <cstring>
#include <cctype>
#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <tuple>
#include <boost/algorithm/string.hpp>
#include "boost/filesystem.hpp"
#include "spline.h"

using namespace boost::filesystem;

string fileName;
string prefix;
double precision = 0.01;
bool header = true;
unsigned numBootstrapRuns = 0;

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
	double totalEnergy = 0;
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (fileName.substr(fileName.length() - 4) != ".log") {
				continue;
			}
		    cout << "Processing " << fileName << endl;
			auto latR = getLatR(fileName);
			double lat = latR.first;
			if (lat < 15 || lat > 240) {
				continue;
			}
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
				double freq = stod(words[1]);
				double en = stod(words[2]);
				totalEnergy += en;
				mode.insert(i, make_tuple(lat, r, freq, en));
				modeNo++;
		    }
			input.close();
		}
	}
	for (unsigned i = 0; i < allModes.size(); i++) {
		double modeEnergy = 0;
		string modeNo = to_string(i + 1);
		ofstream enStream(string("ens") + modeNo + ".csv");
		ofstream freqStream(string("freqs") + modeNo + ".csv");
		for (unsigned j = 0; j < allModes[i].size(); j++) {
			auto dat = allModes[i][j];
			double lat = get<0>(dat);
			double r = get<1>(dat);
			double freq = get<2>(dat);
			double en = get<3>(dat);
			modeEnergy += en;
			if (j > 0 && lat != get<0>(allModes[i][j - 1])) {
				enStream << endl;
				freqStream << endl;
			}
			enStream << lat << " " << r << " " << (en / totalEnergy) << endl;
			freqStream << lat << " " << r << " " << freq << endl;
		}
		enStream.close();
		freqStream.close();
		cout << "Mode " << modeNo << " energy: " << (modeEnergy / totalEnergy) << endl;
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

	if (argc > 2) {
		numBootstrapRuns = stoi(argv[2]);
	}
	vector<double> xs;
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
		n = xs.size();
		//cout << "xStep: " << xStep << ", xRange: " << xRange << endl;
	}

	for (unsigned i = 0; i < numBootstrapRuns; i++) {
		HilbertHuang hh(xs, ys, prefix);
		hh.calculate();
	}
	return EXIT_SUCCESS;
}

