#include "utils.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

using namespace boost;
using namespace boost::filesystem;
namespace utils {

string Utils::ReadTextFile(const string& rFileName) {
    std::ifstream is(rFileName, std::ifstream::binary);
    string str;
    if (is) {
        // get length of file:
        is.seekg(0, is.end);
        int length = is.tellg();
        is.seekg(0, is.beg);

        str.resize(length, ' '); // reserve space
        char* begin = &*str.begin();

        is.read(begin, length);
        is.close();
    }
	return str;
}

bool Utils::IsLetter(char ch) {
    if ((ch >= 65 && ch <= 80) || ch == '_' || (ch >= 97 && ch <= 122)) {
        return true;
    }
    return false;
}

bool Utils::IsDigit(char ch) {
    if (ch >= 48 && ch <= 57) {
        return true;
    }
    return false;
}

bool Utils::IsLetterOrDigit(char ch) {
    return (IsLetter(ch) | IsDigit(ch));
}

bool Utils::IsWhitespace(char ch) {
    if (ch == '\t' || ch == ' ') {
        return true;
    }
    return false;
}

int Utils::Find(const string& rStr, const string& rStrToFind, int startIndex) {
    assert(startIndex >= 0 && startIndex < (int) rStr.length());
    return rStr.find(rStrToFind, startIndex);
}

void Utils::FindAll(const string& rStr, vector<int>& rIndices, const string& rStrToFind, int startIndex) {
    int index = startIndex - 1;
    while ((index = Find(rStr, rStrToFind, index + 1)) >= 0) {
        rIndices.push_back(index);
    }
}

int Utils::FindWhitespace(const string& rStr, int startIndex) {
    int index1 = rStr.find('\t', startIndex);
    int index2 = rStr.find(' ', startIndex);
    if (index1 >= 0 && index1 < index2) {
        return index1;
    }
    return index2;
}

vector<string> Utils::Split(const string& rStr, const string& rSeparator, bool trim) {
    //Object::Dbg(string("Split 0"));
    int index = -1;
    int count = 1;
    while (true) {
        //Object::Dbg(string("Split 1"));
        index = rStr.find(rSeparator, index + 1);
        //Object::Dbg(string("Split 1: ") + index);
        if (index < 0) {
            break;
        }
        count++;
    }
    vector<string> ret_val;
    //string* ret_val = new string[count];
    index = 0;
    //Object::Dbg(string("Split 2"));
    for (int i = 0; i < count; i++) {
        //Object::Dbg(string("Split 3"));
        int new_index = rStr.find(rSeparator, index);
        //Object::Dbg(string("Split 3: ") + new_index);
        string s;
        if (new_index < 0) {
            s = rStr.substr(index);
        }
        else {
            s = rStr.substr(index, new_index - index);
        }
        //Object::Dbg(string("SplitA ") + i + ": " + s);
        if (trim) {
            boost::trim(s);
        }
        //Object::Dbg(string("SplitB ") + i + ": " + s);
        ret_val.push_back(s);
        index = new_index + 1;
    }
    return ret_val;
}

vector<string> Utils::SplitByChars(const string& rStr, const char separators[], bool ignoreEmpty, bool trim) {
	vector<string> retVal;
	boost::split(retVal, rStr, boost::is_any_of(separators), boost::token_compress_on);
	for (vector<string>::iterator it = retVal.begin() ; it != retVal.end(); ++it) {
		if (trim) {
			boost::trim(*it);
		}
		if (ignoreEmpty && (*it).empty()) {
			it->erase();
		}
	}
	return retVal;
}

string Utils::GetProperty(const string& rStr, const string& rPropertyName) {
    size_t index = rStr.find(rPropertyName);
    //Object::Dbg((string) "GetProperty index: " + index);

    if (index >= rStr.length()) {
        return string();
    }
    index += rPropertyName.length();
    //Object::Dbg((string) "GetProperty index_new: " + index);
    if (index >= rStr.length()) {
        return string();
    }
    string str = rStr.substr(index);
    trim(str);
    //Object::Dbg((string) "GetProperty str: " + str);

    if (str[0] != '=') {
        return string();
    }
    str = str.substr(1);
    trim(str);
    //Object::Dbg((string) "GetProperty str: " + str);

    if (str[0] == '"') {
        str = str.substr(1);
        //Object::Dbg((string) "GetProperty str1: " + str);

        index = str.find('"');
        //Object::Dbg((string) "GetProperty index1: " + index);

        assert(index < str.length());
    } else if (rStr[0] == '\'') {
        str = str.substr(1);
        //Object::Dbg((string) "GetProperty str2: " + str);

        index = str.find('\'');
        //Object::Dbg((string) "GetProperty index2: " + index);

        assert(index < str.length());
    } else {
        index = FindWhitespace(str, 0);
        //Object::Dbg((string) "GetProperty index3: " + index);

    }
    return str.substr(0, index);
}

map<string, string> Utils::ReadPropertiesFromStream(istream& input, bool caseSensitive) {
	map<string, string> retVal;
	for (string line; getline(input, line);) {
		//cout << line << endl;
		std::vector<std::string> words;
		boost::split(words, line, boost::is_any_of("="), boost::token_compress_on);
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
			trim(words[0]);
			if (!caseSensitive) {
				to_upper(words[0]);
			}
			trim(words[1]);
			retVal.insert({words[0], words[1]});
		} else {
			cout << "Skipping line, invalid format: " << line << endl;
		}
    }
	return retVal;
}

map<string, string> Utils::ReadPropertiesFromString(const string& rStr, bool caseSensitive) {
	std::istringstream input(rStr);
	map<string, string> retVal = ReadPropertiesFromStream(input, caseSensitive);
	return retVal;
}

map<string, string> Utils::ReadProperties(const string& rFileName, bool caseSensitive) {
	std::ifstream input(rFileName);
	map<string, string> retVal = ReadPropertiesFromStream(input, caseSensitive);
	input.close();
	return retVal;
}

string Utils::FindProperty(const map<string, string>& rProperties, const string& rKey, const string& rDefaultValue, bool caseSensitive) {
	string key(rKey);
	if (!caseSensitive) {
		to_upper(key);
	}
	auto property = rProperties.find(key);
	return property != rProperties.end() ? property->second : rDefaultValue;
}

int Utils::FindIntProperty(const map<string, string>& rProperties, const string& rKey, int defaultValue, bool caseSensitive) {
	string key(rKey);
	if (!caseSensitive) {
		to_upper(key);
	}
	auto property = rProperties.find(key);
	return property != rProperties.end() ? stoi(property->second) : defaultValue;
}

long Utils::FindLongProperty(const map<string, string>& rProperties, const string& rKey, long defaultValue, bool caseSensitive) {
	string key(rKey);
	if (!caseSensitive) {
		to_upper(key);
	}
	auto property = rProperties.find(key);
	return property != rProperties.end() ? stol(property->second) : defaultValue;

}

float Utils::FindFloatProperty(const map<string, string>& rProperties, const string& rKey, float defaultValue, bool caseSensitive) {
	string key(rKey);
	if (!caseSensitive) {
		to_upper(key);
	}
	auto property = rProperties.find(key);
	return property != rProperties.end() ? stof(property->second) : defaultValue;

}

double Utils::FindDoubleProperty(const map<string, string>& rProperties, const string& rKey, double defaultValue, bool caseSensitive) {
	string key(rKey);
	if (!caseSensitive) {
		to_upper(key);
	}
	auto property = rProperties.find(key);
	return property != rProperties.end() ? stod(property->second) : defaultValue;

}

template<> vector<int> Utils::FromStringVector<>(const vector<string>& rVect) {
	vector<int> retVal(rVect.size());
	for (auto s : rVect) {
		retVal.push_back(stoi(s));
	}
	return retVal;
}

template<> vector<unsigned> Utils::FromStringVector<>(const vector<string>& rVect) {
	vector<unsigned> retVal(rVect.size());
	for (auto s : rVect) {
		retVal.push_back(stoi(s));
	}
	return retVal;
}

template<> vector<size_t> Utils::FromStringVector<>(const vector<string>& rVect) {
	vector<size_t> retVal(rVect.size());
	for (auto s : rVect) {
		retVal.push_back(stoi(s));
	}
	return retVal;
}

template<> vector<float> Utils::FromStringVector<>(const vector<string>& rVect) {
	vector<float> retVal(rVect.size());
	for (auto s : rVect) {
		retVal.push_back(stof(s));
	}
	return retVal;
}

template<> vector<double> Utils::FromStringVector<>(const vector<string>& rVect) {
	vector<double> retVal(rVect.size());
	for (auto s : rVect) {
		retVal.push_back(stod(s));
	}
	return retVal;
}

}
