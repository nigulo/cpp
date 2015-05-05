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

void collect(double depth, const string& fileNamePattern) {
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
				if (i++ % 100 != 0) {
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

int main(int argc, char** argv) {
	if (argc == 3) {
		collect(stod(argv[1]), string(argv[2]));
		return EXIT_SUCCESS;
	}
	cout << "Depth and file name prefix not defined" << endl;
	return EXIT_FAILURE;
}
