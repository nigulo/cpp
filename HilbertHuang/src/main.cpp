#include "HilbertHuang.h"
#include "AnalyticSignal.h"
#include "TimeSeries.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <random>
#include <memory>

using namespace boost::filesystem;

string fileName;
string prefix;
double precision = 0.01;
bool header = true;
unsigned numBootstrapRuns = 1;
double noisePercent = 0;

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
	if (!exists(fileName)) {
		cout << "Input file not found" << endl;
		return EXIT_FAILURE;
	}
    cout << "Processing " << fileName << endl;
	string::size_type n = fileName.find('.');
	prefix = fileName.substr(0, n);

	// Check if IMF-s already calculated
	if (exists(prefix + ".log")) {
		return EXIT_SUCCESS;
	}

	if (argc > 2) {
		numBootstrapRuns = stoi(argv[2]);
		noisePercent = 0.1;
	}

	if (argc > 3) {
		noisePercent = stod(argv[3]);
	}

	TimeSeries ts;
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
				ts.add(xVal, yVal);
			} catch (invalid_argument& ex) {
				cout << "Skipping line, invalid number: " << line << endl;
			}
		} else {
			cout << "Skipping line, invalid number of columns: " << line << endl;
		}
    }
	input.close();

	auto noiseStdDev = sqrt(ts.meanVariance().second) * noisePercent;
	vector<TimeSeries> ensemble;
    random_device rd;
	default_random_engine e1(rd());
	normal_distribution<double> dist(0, noiseStdDev);
	for (unsigned i = 0; i < numBootstrapRuns; i++) {
		TimeSeries* ts1 = new TimeSeries(ts);
		if (noisePercent > 0) {
			*ts1 = *ts1 + dist(e1);
		}
		HilbertHuang hh(ts1);
		hh.calculate();
		const vector<unique_ptr<TimeSeries>>& imfs = hh.getImfs();
		for (unsigned i = 0; i < imfs.size(); i++) {
			if (i >= ensemble.size()) {
				ensemble.push_back(*imfs[i].get());
			} else {
				ensemble[i] = ensemble[i] + *imfs[i].get();
			}
		}
	}
	stringstream logText;
	int modeNo = 1;
	for (auto i = ensemble.begin(); i != ensemble.end(); i++) {
		TimeSeries& imf = (*i) / numBootstrapRuns;
		int numZeroCrossings = imf.findNumZeroCrossings();
		double xRange = *(imf.getXs().end() - 1) - *(imf.getXs().begin());
		double meanFreq = 0.5 * numZeroCrossings / xRange;
		double meanEnergy = AnalyticSignal::calculate(imf, modeNo, prefix);
        logText << modeNo << ": " << meanFreq << " " << meanEnergy << endl;
        modeNo++;
	}
	ofstream logStream(prefix + ".log");
	logStream << logText.str();
	logStream.close();
	cout << logText.str();
	return EXIT_SUCCESS;
}

