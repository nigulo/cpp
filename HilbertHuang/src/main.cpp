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

#define MIN_LAT 15
#define MAX_LAT 240

#define DELTA_R 10
#define BOT 	10
#define MID 	64
#define SURF 	120

#define EQUATOR 127.5

void collect() {
	vector<vector<tuple<double, double, double, double, double>>> allModes;
	directory_iterator end_itr; // default construction yields past-the-end
	path currentDir(".");
	double totalEnergy = 0;
	double totalEnergyBot = 0;
	double totalEnergyMid = 0;
	double totalEnergySurf = 0;
	double totalVar = 0;
	double totalVarBot = 0;
	double totalVarMid = 0;
	double totalVarSurf = 0;
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (fileName.substr(fileName.length() - 4) != ".log") {
				continue;
			}
		    cout << "Processing " << fileName << endl;
			auto latR = getLatR(fileName);
			double lat = latR.first;
			if (lat < MIN_LAT || lat > MAX_LAT) {
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
					allModes.push_back(vector<tuple<double, double, double, double, double>>());
				}
				vector<tuple<double, double, double, double, double>>& mode = allModes[modeNo];
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
				double var = stod(words[3]);
				totalEnergy += en;
				totalVar += var;
				if (abs(r - BOT) < DELTA_R) {
					totalEnergyBot += en;
					totalVarBot += var;
				}
				if (abs(r - MID) < DELTA_R) {
					totalEnergyMid += en;
					totalVarMid += var;
				}
				if (abs(r - SURF) < DELTA_R) {
					totalEnergySurf += en;
					totalVarSurf += var;
				}
				mode.insert(i, make_tuple(lat, r, freq, en, var));
				modeNo++;
		    }
			input.close();
		}
	}
	cout << "Mode energies (total bot mid surf)" << endl;
	for (unsigned i = 0; i < allModes.size(); i++) {
		double modeEnergy = 0;
		double modeEnergyBot = 0;
		double modeEnergyMid = 0;
		double modeEnergySurf = 0;
		double modeVar = 0;
		double modeVarBot = 0;
		double modeVarMid = 0;
		double modeVarSurf = 0;
		double modeFreqSum = 0;
		double modeWeightSum = 0;
		double maxEnergyLatN = EQUATOR;
		double maxEnergyLatS = EQUATOR;
		double maxEnergyRN = 120;
		double maxEnergyRS = 120;
		double maxEnergyN = 0;
		double maxEnergyS = 0;
		string modeNo = to_string(i + 1);
		ofstream enStream(string("ens") + modeNo + ".csv");
		ofstream freqStream(string("freqs") + modeNo + ".csv");
		for (unsigned j = 0; j < allModes[i].size(); j++) {
			auto dat = allModes[i][j];
			double lat = get<0>(dat);
			double r = get<1>(dat);
			double freq = get<2>(dat);
			double en = get<3>(dat);
			double var = get<4>(dat);
			modeEnergy += en;
			modeVar += var;
			modeFreqSum += en * freq;
			modeWeightSum += en;
			if (abs(r - BOT) < DELTA_R) {
				modeEnergyBot += en;
				modeVarBot += var;
			}
			if (abs(r - MID) < DELTA_R) {
				modeEnergyMid += en;
				modeVarMid += var;
			}
			if (abs(r - SURF) < DELTA_R) {
				modeEnergySurf += en;
				modeVarSurf += var;
			}
			if (j > 0 && lat != get<0>(allModes[i][j - 1])) {
				enStream << endl;
				freqStream << endl;
			}
			if (lat > EQUATOR && en > maxEnergyN) {
				maxEnergyN = en;
				maxEnergyLatN = lat;
				maxEnergyRN = r;
			} else if (lat < EQUATOR && en > maxEnergyS) {
				maxEnergyS = en;
				maxEnergyLatS = lat;
				maxEnergyRS = r;
			}
			enStream << lat << " " << r << " " << (en / totalEnergy) << endl;
			freqStream << lat << " " << r << " " << freq << endl;
		}
		enStream.close();
		freqStream.close();
		double modeFreqMean = modeFreqSum / modeWeightSum;
		double modeFreqVar = 0;
		for (unsigned j = 0; j < allModes[i].size(); j++) {
			auto dat = allModes[i][j];
			double freq = get<2>(dat);
			double en = get<3>(dat);
			modeFreqVar += en * (freq - modeFreqMean) * (freq - modeFreqMean);
		}
		modeFreqVar /= modeWeightSum;
		cout << modeNo << ": " << modeFreqMean << " " << sqrt(modeFreqVar) << " "
				<< (modeEnergy / totalEnergy) << " "
				<< (modeEnergyBot / totalEnergyBot)  << " "
				<< (modeEnergyMid / totalEnergyMid) << " "
				<< (modeEnergySurf / totalEnergySurf)
				<< (modeVar / totalVar) << " "
				<< (modeVarBot / totalVarBot)  << " "
				<< (modeVarMid / totalVarMid) << " "
				<< (modeVarSurf / totalVarSurf)
				<< " (" << maxEnergyLatN << ", " << maxEnergyRN  << " - "
				<< maxEnergyLatS << ", " << maxEnergyRS << ")" << endl;
	}
}

TimeSeries* loadTimeSeries(const string& fileName) {
	auto ts = new TimeSeries();
	ifstream input(fileName);
	for (string line; getline(input, line);) {
		//cout << line << endl;
		std::vector<std::string> words;
		boost::split(words, line, boost::is_any_of("\t "), boost::token_compress_on);
		for (vector<string>::iterator it = words.begin(); it != words.end();) {
			//cout << "<" << (*it) << ">" << endl;
			if ((*it).length() == 0) {
				it = words.erase(it);
			} else {
				it++;
			}
		}
		if (words.size() > 0 && words[0][0] == '#') {
			//cout << "Skipping comment line: " << line << endl;
		} else if (words.size() == 2) {
			try {
				double xVal = stod(words[0]);
				double yVal = stod(words[1]);
				ts->add(xVal, yVal);
			} catch (invalid_argument& ex) {
				cout << "Skipping line, invalid number: " << line << endl;
			}
		} else {
			cout << "Skipping line, invalid number of columns: " << line << endl;
		}
    }
	input.close();
	return ts;
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

	auto ts = loadTimeSeries(fileName);
	TimeSeries ts2(*ts);
	vector<TimeSeries> ensemble;

	// Check if IMF-s already calculated
	if (exists(prefix + ".log")) {
		cout << "Loading IMF-s" << endl;
		//return EXIT_SUCCESS;
		int modeNo = 0;
		for (;;) {
			string imfFileName(prefix + "_imf_" + to_string(++modeNo) + ".csv");
			if (!exists(imfFileName)) {
				cout << to_string(modeNo - 1) << " IMF-s loaded" << endl;
				break;
			}
			ensemble.push_back(*loadTimeSeries(imfFileName));
		}
	} else {
		cout << "Computing IMF-s" << endl;
		if (argc > 2) {
			numBootstrapRuns = stoi(argv[2]);
			noisePercent = 0.1;
		}

		if (argc > 3) {
			noisePercent = stod(argv[3]);
		}

		auto noiseStdDev = sqrt(ts->meanVariance().second) * noisePercent;
		random_device rd;
		default_random_engine e1(rd());
		normal_distribution<double> dist(0, noiseStdDev);
		for (unsigned i = 0; i < numBootstrapRuns; i++) {
			TimeSeries* ts1 = new TimeSeries(*ts);
			if (noisePercent > 0) {
				for (ts1->begin(); ts1->hasNext(); ts1->next()) {
					ts1->setY(ts1->getY() + dist(e1));
				}
			}
			HilbertHuang hh(ts1);
			hh.calculate();
			const vector<unique_ptr<TimeSeries>>& imfs = hh.getImfs();
			for (unsigned i = 0; i < imfs.size(); i++) {
				if (i >= ensemble.size()) {
					ensemble.push_back(*imfs[i]);
				} else {
					ensemble[i] + *imfs[i];
				}
				// Reconstruction error check
				//*ts1 + *imfs[i];
			}
			//ts2 - *ts1;
			//auto meanVar = ts2.meanVariance();
			//cout << "Residue variance: " << meanVar.second << endl;
		}
	}
	stringstream logText;
	int modeNo = 1;
	double var = ts2.meanVariance().second;
	for (auto i = ensemble.begin(); i != ensemble.end(); i++) {
		TimeSeries& imf = (*i) / numBootstrapRuns;
		int numZeroCrossings = imf.findNumZeroCrossings();
		double xRange = *(imf.getXs().end() - 1) - *(imf.getXs().begin());
		double meanFreq = 0.5 * numZeroCrossings / xRange;
		double meanEnergy = AnalyticSignal::calculate(imf, modeNo, prefix);
		ts2 - imf;
		double newVar = ts2.meanVariance().second;
        logText << modeNo << ": " << meanFreq << " " << meanEnergy << " " << (var - newVar) << endl;
        var = newVar;
        modeNo++;
	}
	ofstream logStream(prefix + ".log");
	logStream << logText.str();
	logStream.close();
	cout << logText.str();
	return EXIT_SUCCESS;
}

