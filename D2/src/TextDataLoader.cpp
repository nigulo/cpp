#include "TextDataLoader.h"
#include <boost/algorithm/string.hpp>
#include <iostream>

TextDataLoader::TextDataLoader(const string& fileName, unsigned bufferSize,
		const vector<unsigned>& dims,
		const vector<unsigned>& mins,
		const vector<unsigned>& maxs,
		unsigned totalNumVars, const vector<unsigned>& varIndices) :
				DataLoader(fileName, bufferSize, ios::in, dims, mins, maxs, totalNumVars, varIndices) {
}

TextDataLoader::TextDataLoader(const TextDataLoader& dataLoader) : DataLoader(dataLoader) {
	for (int i = 0; i <= dataLoader.page; i++) {
		Next();
	}
}

TextDataLoader::~TextDataLoader() {
}

bool TextDataLoader::Next() {
	if (!input.is_open()) {
		return false;
	}
	page++;
	delete[] data;
	data = new real[bufferSize * (dim * totalNumVars + 1)];
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
			} else if (words.size() >= dim * totalNumVars + 1) {
				try {
					data[i * (dim * totalNumVars + 1)] = stod(words[0]); // x
					for (unsigned j = 0; j < dim * totalNumVars; j++) {
						data[i * (dim * totalNumVars + 1) + j + 1] = stod(words[j + 1]);
					}
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
	pageSize = i;
	return pageSize > 0;
}

DataLoader* TextDataLoader::Clone() const {
	return new TextDataLoader(*this);
}
