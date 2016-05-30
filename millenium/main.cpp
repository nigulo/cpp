#include <fstream>
#include <sstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <random>
#include <memory>
#include <cstdint>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <fftw3.h>
#include "utils/utils.h"

using namespace boost::filesystem;
using namespace boost;
using namespace std;
using namespace utils;

#define sgn(a) (a > 0 ? 1 : a < 0 ? -1 : 0)

void fft(bool direction, int n, fftw_complex* data) {
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);

    memcpy(in, data, sizeof(fftw_complex) * n);
	fftw_plan p = fftw_plan_dft_1d(n, in, out, direction ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */
    memcpy(data, out, sizeof(fftw_complex) * n);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

}

string getPrefix(const string& fileName) {
	string prefix;
	for (unsigned i = 0; i < fileName.size(); i++) {
		if (isdigit(fileName[i])) {
			prefix = fileName.substr(0, i);
			break;
		}
	}
	return prefix;
}

#define EQUATOR 127.5f
#define R_GRID 128
#define LAT_GRID 256
#define LAT_SPAN 150.0f

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

// returns co-latitude and radius in physical coordinates
pair<double, double> simToPhysCoord(const pair<double, double>& latR) {
	return {90 - (EQUATOR - latR.first) * LAT_SPAN / LAT_GRID, 0.7 + latR.second * 0.3 / R_GRID};
}

/**
 * Collects the mode data at given depth and outputs it as a coloured map
 */
void collect(double depth, const string& fileNamePattern, unsigned sampling) {
	vector<tuple<double, double, double>> xyzs;
	directory_iterator end_itr; // default construction yields past-the-end
	path currentDir(".");
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (Utils::Find(fileName, fileNamePattern) < 0) {
				continue;
			}
		    cout << "Processing " << fileName << endl;
			auto latR = getLatR(fileName);
			double lat = latR.first;
			if (lat < 15 || lat > 240) {
				continue;
			}
			double r = latR.second;
			if (r != depth) {
				continue;
			}
			ifstream input(fileName);
			int i = 0;
			for (string line; getline(input, line);) {
				if (i++ % sampling != 0) {
					continue;
				}
				//cout << line << endl;
				std::vector<std::string> words;
				boost::split(words, line, boost::is_any_of("\t "), boost::token_compress_on);
				for (vector<string>::iterator it = words.begin() ; it != words.end(); ++it) {
					if ((*it).length() == 0) {
						words.erase(it);
					}
				}
				double x = stod(words[0]);
				double z = stod(words[1]);
				auto i = xyzs.begin();
				for (; i != xyzs.end(); i++) {
					if (get<0>(*i) > x || (get<0>(*i) == x && get<1>(*i) > lat)) {
						break;
					}
				}
				xyzs.insert(i, make_tuple(x, lat, z));
		    }
			input.close();
		}
	}
	ofstream output(string("butterfly.csv"));
	double xPrev = get<0>(xyzs[0]);
	for (auto&& xyz : xyzs) {
		if (get<0>(xyz) > xPrev) {
			xPrev = get<0>(xyz);
			output << endl;
		}
		output << get<0>(xyz) << " " << get<1>(xyz) << " " << get<2>(xyz) << endl;
	}
	output.close();
}

/**
 * Calculates the mean parity of the mode
 */
void parity(uint8_t mode) {
	vector<double> ts;
	vector<pair<double /*even*/, double/*odd*/ >> parity;
	directory_iterator end_itr; // default construction yields past-the-end
	string fileNamePattern=string("imf_") + to_string(mode) + ".csv";
	path currentDir(".");
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (Utils::Find(fileName, fileNamePattern) < 0) {
				continue;
			}
			auto latR = getLatR(fileName);
			string prefix = getPrefix(fileName);
			int latSim = latR.first;
			if (latSim < 15 || latSim > 240) { // If we want to omit boundary regions
				continue;
			}
			if (latSim > EQUATOR) {
				continue;
			}
		    //cout << "Processing " << fileName << endl;
			int rSim = latR.second;
			auto latRPhys = simToPhysCoord(latR);
			double th = latRPhys.first * M_PI / 180;
			double r = latRPhys.second;
			//cout << "r, th: " << rSim << " " << r << ", " << latSim << " " << th * 180 / M_PI << endl;
			vector<double> zsNorth;
			ifstream inputNorth(fileName);
			for (string line; getline(inputNorth, line);) {
				//cout << line << endl;
				std::vector<std::string> words;
				boost::split(words, line, boost::is_any_of("\t "), boost::token_compress_on);
				for (vector<string>::iterator it = words.begin() ; it != words.end(); ++it) {
					if ((*it).length() == 0) {
						words.erase(it);
					}
				}
				double x = stod(words[0]);
				double zNorth = stod(words[1]);
				ts.push_back(x);
				zsNorth.push_back(zNorth);
		    }
			inputNorth.close();
			if (parity.size() < zsNorth.size()) {
				parity.resize(zsNorth.size(), {0, 0});
			}
			int oppositeLat = 2 * EQUATOR - latSim;
			ifstream inputSouth(prefix + to_string(oppositeLat) + "_" + to_string(rSim) + "_imf_" + to_string(mode) + ".csv");
			//cout << (prefix + to_string(oppositeLat) + "_" + to_string(r) + "_imf_" + to_string(mode) + ".csv") << endl;
			int i = 0;
			for (string line; getline(inputSouth, line);) {
				//cout << line << endl;
				std::vector<std::string> words;
				boost::split(words, line, boost::is_any_of("\t "), boost::token_compress_on);
				for (vector<string>::iterator it = words.begin() ; it != words.end(); ++it) {
					if ((*it).length() == 0) {
						words.erase(it);
					}
				}
				//double x = stod(words[0]);
				double zSouth = stod(words[1]);
				double zNorth = zsNorth[i];
				double even = 0.5 * (zNorth + zSouth);
				double odd = 0.5 * (zNorth - zSouth);
				double r2sinth = r * r * sin(th);
				parity[i].first += r2sinth * even * even;
				parity[i].second += r2sinth * odd * odd;
				//cout << parity[i].first << ", " << parity[i].second << endl;
				i++;
		    }
			inputSouth.close();
		}
	}
	ofstream output(string("parity_") + to_string((int) mode) + ".csv");
	double total = 0;
	size_t i = 0;
	for (auto p : parity) {
		double val = (p.first - p.second) / (p.first + p.second);
		total += val;
		output << ts[i++] << " " << val << endl;
	}
	output.close();
	cout << "Mean parity for mode " << ((int) mode) << ": " << total / parity.size() << endl;
}

//#define LATITUDE_FILTER

/**
 * Calculates the mean spectral entropy of the mode
 */
void entropy(uint8_t mode) {
	double totalEntropy = 0;
	double totalEnergy = 0;
	size_t count = 0;
	directory_iterator end_itr; // default construction yields past-the-end
	string fileNamePattern=string("imf_") + to_string(mode) + ".csv";
	path currentDir(".");
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (Utils::Find(fileName, fileNamePattern) < 0) {
				continue;
			}
			string prefix = getPrefix(fileName);
#ifdef LATITUDE_FILTER
			auto latR = getLatR(fileName);
			int lat = latR.first;
			if (lat < 15 || lat > 240) { // If we want to omit boundary regions
				continue;
			}
#endif
		    //cout << "Processing " << fileName << endl;
			//int r = latR.second;
			vector<double> zs;
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
				//double x = stod(words[0]);
				double z = stod(words[1]);
				zs.push_back(z);
		    }
			input.close();
			const auto n = zs.size();
		    fftw_complex* data = (fftw_complex*) malloc(sizeof(fftw_complex) * n);
		    for (size_t i = 0; i < n; i++) {
		    	data[i][0] = zs[i];
		    	data[i][1] = 0;
		    }
			fft(true, n, data);
			double powerSpec[n];
			double norm = 0;
		    double energy = 0;
		    for (size_t i = 0; i < n; i++) {
		    	powerSpec[i] = data[i][0] * data[i][0] + data[i][1] * data[i][1];
		    	norm += powerSpec[i] + 1e-12;
		    	energy += powerSpec[i];
		    }
		    free(data);
		    double entropy = 0;
		    for (size_t i = 0; i < n; i++) {
		    	double d = powerSpec[i] / norm;
		    	entropy -= d * log2(d + 1e-12);
		    }
		    entropy /= log2(n);
		    totalEntropy += entropy;
		    totalEnergy += energy;
		    count++;
		}
	}
	cout << "Mean entropy, energy for mode " << ((int) mode) << ": " << totalEntropy / count << ", " << totalEnergy / count << endl;
}

double getLat(const string& fileName) {
	int index = fileName.find('_');
	index = fileName.find('_', index + 1);
	string latStr = fileName.substr(index + 1);
	double lat = -1;
	for (unsigned i = 0; i < latStr.length(); i++) {
		if (!isdigit(latStr[i])) {
			lat = stod(latStr.substr(0, i));
			break;
		}
	}
	return lat;
}


void wings(bool nOrS) {
	double minLat = 19;
	double maxLat = 31;
	double threshold = 4;
	vector<double> ts;
	vector<tuple<double /*minLat*/, double /*maxLat*/>> wings;
	vector<double> cycleStrengths;
	directory_iterator end_itr; // default construction yields past-the-end
	path currentDir(".");
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (fileName.substr(fileName.length() - 4) != ".txt") {
				continue;
			}
		    cout << "Processing " << fileName << endl;
			double lat = getLat(fileName);
			if (nOrS && lat > 127.5) {
				continue;
			}
			if (!nOrS && lat < 127.5) {
				continue;
			}
			if (lat < 15 || lat > 240) {
				continue;
			}
			lat = (lat - 127.5) * 150 / 255;
			if (abs(lat) < minLat || abs(lat) > maxLat) {
				continue;
			}
			ifstream input(fileName);
			unsigned i = 0;
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
						double t = stod(words[0]);
						double b = stod(words[1]);
						if (wings.size() <= i) {
							ts.push_back(t);
							wings.push_back(make_tuple(90, 0));
						}
						auto wing = wings[i];
						if (abs(b) >= threshold) {
							if (abs(lat) < abs(get<0>(wing))) {
								wings[i] = make_tuple(lat, get<1>(wing));
							} else if (abs(lat) > abs(get<1>(wing))) {
								wings[i] = make_tuple(get<0>(wing), lat);
							}
						}
						i++;
					} catch (invalid_argument& ex) {
						cout << "Skipping line, invalid number: " << line << endl;
					}
				} else {
					cout << "Skipping line, invalid number of columns: " << line << endl;
				}
		    }
			input.close();
		}
	}
	ofstream output(string("wings.csv"));
	for (unsigned i = 0; i < ts.size(); i++) {
		if (abs(get<0>(wings[i])) < 90 && abs(get<1>(wings[i])) > 0) {
			output << ts[i] << " " << get<0>(wings[i]) << " " << get<1>(wings[i]) << " " << (get<1>(wings[i]) - get<0>(wings[i])) << endl;
		}
	}
	output.close();
}

enum Component {
	PHI,
	R,
};

enum Hemisphere {
	N,
	S,
	BOTH
};

void cycles(Component component, Hemisphere hemisphere, double cycleLengthLimit, bool detectCycleLength) {
	double minLat = 19;
	double maxLat = 31;
	if (component == R) {
		minLat = 63;
		maxLat = 75;
	}
	vector<double> cycleStrengths;
	vector<double> cycleLengths;
	directory_iterator end_itr; // default construction yields past-the-end
	path currentDir(".");
	for (directory_iterator itr(currentDir); itr != end_itr; ++itr) {
		if (is_regular_file(itr->status())) {
			const string& fileName = itr->path().generic_string();
			if (fileName.substr(fileName.length() - 4) != ".txt") {
				continue;
			}
		    //cout << "Processing " << fileName << endl;
			double lat = getLat(fileName);
			if (hemisphere == N && lat > 127.5) {
				continue;
			}
			if (hemisphere == S && lat < 127.5) {
				continue;
			}
			//if (lat < 15 || lat > 240) {
			//	continue;
			//}
			lat = (lat - 127.5) * 150 / 255;
			if (abs(lat) < minLat || abs(lat) > maxLat) {
				continue;
			}
			ifstream input(fileName);
			double cycleStart = 0;
			double bLast = -1;
			double bInt = 0;
			int numCycles = 0;
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
						double t = stod(words[0]);
						double b = stod(words[1]);
						if (bLast < 0) {
							bLast = b;
						}
						bInt += b * b;

						if ((t - cycleStart) >= cycleLengthLimit && (!detectCycleLength || sgn(b) != sgn(bLast))) {
							cycleStrengths.push_back(bInt / (t - cycleStart));
							cycleLengths.push_back(t - cycleStart);
							bInt = 0;
							cycleStart = t;
							numCycles++;
						}
						bLast = b;
					} catch (invalid_argument& ex) {
						cout << "Skipping line, invalid number: " << line << endl;
					}
				} else {
					cout << "Skipping line, invalid number of columns: " << line << endl;
				}
		    }
			cout << "lat: " << lat << ", numCycles: " << numCycles << endl;
			input.close();
		}
	}
	string comp(component == PHI ? "_phi" : "_r");
	string hem(hemisphere == N ? "_N" : (hemisphere == S ? "_S" : ""));
	ofstream output(string("cycles")+ comp + hem + ".csv");
	for (size_t i = 0; i < cycleStrengths.size(); i++) {
		output << cycleStrengths[i] << " " << cycleLengths[i] << endl;
	}
	output.close();
}

int main(int argc, char** argv) {
	if (argc  <= 1) {
		cout << "Usage: millenium options\n    options: wings|collect|parity|cycles" << endl;
		return EXIT_FAILURE;
	}
	string option(argv[1]);
	to_upper(option);
	if (option == "WINGS" && argc >= 3) {
		wings(string(argv[2]) == "N" || string(argv[2]) == "n");
		return EXIT_SUCCESS;
	} else if (option == "MAP" && argc >= 3) {
		unsigned sampling = 100;
		if (argc >= 5) {
			sampling = stoi(argv[4]);
		}
		collect(stod(argv[2]), string(argv[3]), sampling);
		return EXIT_SUCCESS;
	} else if (option == "PARITY" && argc >= 3) {
		uint8_t minMode = stoi(argv[2]);
		uint8_t maxMode = minMode;
		if (argc >= 4) {
			maxMode = stoi(argv[3]);
		}
		for (uint8_t mode = minMode; mode <= maxMode; mode++) {
			parity(mode);
		}
		return EXIT_SUCCESS;
	} else if (option == "ENTROPY" && argc >= 3) {
		uint8_t minMode = stoi(argv[2]);
		uint8_t maxMode = minMode;
		if (argc >= 4) {
			maxMode = stoi(argv[3]);
		}
		for (uint8_t mode = minMode; mode <= maxMode; mode++) {
			entropy(mode);
		}
		return EXIT_SUCCESS;
	} else if (option == "CYCLES" && argc >= 3) {
		string component(argv[2]);
		to_upper(component);
		Hemisphere hemisphere(BOTH);
		if (argc >= 4) {
			string hem(argv[3]);
			to_upper(hem);
			hemisphere = hem == "N" ? N : (hem == "S" ? S : BOTH);
		}
		double cycleLengthLimit = 1;
		if (argc >= 5) {
			cycleLengthLimit = stof(argv[4]);
		}
		bool detectCycleLength = true;
		if (argc >= 6) {
			detectCycleLength = stoi(argv[5]) == 1;
		}
		cycles(component == "PHI" ? PHI : R, hemisphere, cycleLengthLimit, detectCycleLength);
		return EXIT_SUCCESS;
	} else {
		cout << "Usage: millenium options\n    options: wings|map|parity|cycles" << endl;
		return EXIT_FAILURE;
	}
}
