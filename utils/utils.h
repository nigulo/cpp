
#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>

using namespace std;
namespace utils {

class Utils {
public:
	static string ReadTextFile(const string& rFileName);

	// string utils
	static bool IsLetterOrDigit(char ch);
    static bool IsLetter(char ch);
    static bool IsDigit(char ch);
    static bool IsWhitespace(char ch);

    /**
     * Removes all leading and trailing whitespaces from this string
     * @param rStr string to trim
     * @return reference to this string
     **/
    static string& Trim(string& rStr);

    /**
     * Searches for a substring in this string from the
     * given start index.
     * @param rStr string to search from
     * @param rStrToFind substring to find
     * @param startIndex start index or 0 if omitted
     * @return index of the first character of the substring
     * or -1 if substring is not found
     **/
    static int Find(const string& rStr, const string& rStrToFind, int startIndex = 0);

    /**
     * Searches for all occurances of the specified substring
     * starting from the given start index.
     * @param rStr string to search from
     * @param rIndices the found indices are inserted here
     * @param rStrToFind substring to find
     * @param startIndex start index or 0 if omitted
     **/
    static void FindAll(const string& rStr, vector<int>& rIndices, const string& rStrToFind, int startIndex = 0);

    /**
     * Finds next whitespace character starting from the
     * given position
     * @param rStr string to search from
     * @return index of the whitespace character or -1 if
     * not found
     **/
    static int FindWhitespace(const string& rStr, int startIndex = 0);

    /**
     * Splits the string to an array using the given separator
     * @param rStr string to split
     * @param rSeparator separator
     * @param trim specifies whether to trim the chunks after splitting
     * @return list of chunks
     **/
    static vector<string> Split(const string& rStr, const string& rSeparator, bool trim = true);

    /**
     * Returns the property with the given name cointained in the
     * string. Properties are defined in the string as "name = value"
     * pairs. The value may be put between quotation marks to allow
     * spaces.
     * If no property is found empty string is returned
     * @param rStr string to search from
     * @param rPropertyName name of property to search
     **/
    static string GetProperty(const string& rStr, const string& rPropertyName);

};

} /* namespace utils */

#endif /* UTILS_H_ */
