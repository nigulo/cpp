#include "utils.h"
#include <fstream>

using namespace utils;

std::string Utils::ReadTextFile(const std::string& rFileName) {
    std::ifstream is(rFileName, std::ifstream::binary);
    std::string str;
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
