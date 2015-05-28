#include <fstream>
#include <sstream>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <random>
#include <memory>
#include "utils/utils.h"

using namespace boost::filesystem;
using namespace std;
using namespace utils;

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
			vector<double[4]> modes;
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

double minLat = 19;
double maxLat = 31;
double threshold = 4;

void wings(bool nOrS) {
	vector<double> ts;
	vector<tuple<double /*minLat*/, double /*maxLat*/>> wings;
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
			if (nOrS && lat < 127.5) {
				continue;
			}
			if (!nOrS && lat > 127.5) {
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

int main(int argc, char** argv) {
	if (argc == 2) {
		wings(string(argv[1]) == "N" || string(argv[1]) == "n");
		return EXIT_SUCCESS;
	}
	if (argc >= 3) {
		unsigned sampling = 100;
		if (argc >= 4) {
			sampling = stoi(argv[3]);
		}
		collect(stod(argv[1]), string(argv[2]), sampling);
		return EXIT_SUCCESS;
	}
	cout << "Depth and file name prefix not defined" << endl;
	return EXIT_FAILURE;
}
