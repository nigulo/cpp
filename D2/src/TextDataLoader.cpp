#include "TextDataLoader.h"
#include <boost/algorithm/string.hpp>

TextDataLoader::TextDataLoader(const string& fileName, unsigned bufferSize, ios::openmode mode, unsigned dim, unsigned totalNumVars, const vector<unsigned>& varIndices) :
		DataLoader(fileName, bufferSize, ios::in, dim, totalNumVars, varIndices) {
}

TextDataLoader::TextDataLoader(const TextDataLoader& dataLoader) : DataLoader(dataLoader) {
	for (int i = 0; i < dataLoader.page; i++) {
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
			} else if (words.size() == dim * totalNumVars) {
				try {
					double xVal = stod(words[0]);
					x.push_back(xVal);
					double* yVals = new double[dim * varIndices.size()];
					for (unsigned j = 0; j < dim; j++) {
						unsigned k = 0;
						for (unsigned varIndex : varIndices) {
							yVals[j * dim + k++] = stod(words[dim * totalNumVars + varIndex]);
						}
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

unique_ptr<DataLoader> TextDataLoader::Clone() const {
	return unique_ptr<DataLoader>(new TextDataLoader(*this));
}
