/*
 * DataLoader.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: nigul
 */

#include "DataLoader.h"
#include <boost/algorithm/string.hpp>

DataLoader::DataLoader(const string& fileName, unsigned startCol, unsigned dim, unsigned bufferSize) :
	fileName(fileName),
	input(fileName),
	startCol(startCol),
	dim(dim),
	bufferSize(bufferSize),
	page(-1)
{
}

DataLoader::DataLoader(const DataLoader& dataLoader) :
	fileName(dataLoader.fileName),
	input(fileName),
	startCol(dataLoader.startCol),
	dim(dataLoader.dim),
	bufferSize(dataLoader.bufferSize),
	page(-1)
{
	for (int i = 0; i < dataLoader.page; i++) {
		Next();
	}
}

DataLoader::~DataLoader() {
	if (input.is_open()) {
		input.close();
	}
}

bool DataLoader::Next() {
	if (!input.is_open()) {
		return false;
	}
	page++;
	x.clear();
	for (auto& yVals : y) {
		delete[] yVals;
	}
	y.clear();
	unsigned i = 0;
	while (bufferSize == 0 || i < bufferSize) {
		string line;
		if (getline(input, line)) {
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
			} else if (words.size() > startCol) {
				try {
					double xVal = stod(words[0]);
					x.push_back(xVal);
					double* yVals = new double[dim];
					for (unsigned i = 0; i < dim; i++) {
						yVals[i] = stod(words[startCol + i]);
					}
					y.push_back(yVals);
				} catch (std::invalid_argument& ex) {
					cout << "Skipping line, invalid number: " << line << endl;
				}
			} else {
				cout << "Skipping line, too few columns: " << line << endl;
			}
			i++;
		} else {
			input.close();
			break;
		}
	}
	return !x.empty();
}

void DataLoader::Reset() {
	x.clear();
	for (auto& yVals : y) {
		delete[] yVals;
	}
	y.clear();
	if (input.is_open()) {
		input.close();
	}
	input.open(fileName);
	page = -1;
}
