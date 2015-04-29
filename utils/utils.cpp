#include "utils.h"
#include <fstream>
#include <cassert>

using namespace utils;

string Utils::ReadTextFile(const string& rFileName) {
    ifstream is(rFileName, ifstream::binary);
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
    assert(startIndex >= 0 && startIndex < rStr.length());
    return rStr.find(rStrToFind, startIndex);
}

void Utils::FindAll(const string& rStr, vector<int>& rIndices, const string& rStrToFind, int startIndex) {
    int index = startIndex - 1;
    while ((index = Find(rStr, rStrToFind, index + 1)) >= 0) {
        rIndices.push_back(index);
    }
}

string& Utils::Trim(string& rStr) {
    int startIndex = 0;
    while (startIndex < rStr.length() && IsWhitespace(rStr[startIndex])) {
        startIndex++;
    }
    int endIndex = rStr.length() - 1;
    while (endIndex > 0 && IsWhitespace(rStr[endIndex])) {
        endIndex--;
    }
    if (endIndex < startIndex) {
    	rStr.clear();
    }
    else {
    	rStr.assign(rStr.substr(startIndex, endIndex - startIndex + 1));
    }
    return rStr;
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
    vector<string> ret_val(count);
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
            Trim(s);
        }
        //Object::Dbg(string("SplitB ") + i + ": " + s);
        ret_val.push_back(s);
        index = new_index + 1;
    }
    return ret_val;
}

string Utils::GetProperty(const string& rStr, const string& rPropertyName) {
    int index = rStr.find(rPropertyName);
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
    Trim(str);
    //Object::Dbg((string) "GetProperty str: " + str);

    if (str[0] != '=') {
        return string();
    }
    str = str.substr(1);
    Trim(str);
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

