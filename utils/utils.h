
#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>
#include <map>

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
     * Splits the string to an array using the separator chars
     * @param rStr string to split
     * @param separators separator chars
     * @param trim specifies whether to trim the chunks after splitting
     * @return list of chunks
     **/
    static vector<string> SplitByChars(const string& rStr, const char separators[], bool ignoreEmpty = true, bool trim = true);

    template<typename T> static vector<T> FromStringVector(const vector<string> & rVect);

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

    static map<string, string> ReadProperties(const string& rFileName, bool caseSensitive = false);
    static string FindProperty(const map<string, string>& rProperties, const string& key, const string& rDefaultValue = string(), bool caseSensitive = false);
    static int FindIntProperty(const map<string, string>& rProperties, const string& key, int defaultValue, bool caseSensitive = false);
    static long FindLongProperty(const map<string, string>& rProperties, const string& key, long defaultValue, bool caseSensitive = false);
    static float FindFloatProperty(const map<string, string>& rProperties, const string& key, float defaultValue, bool caseSensitive = false);
    static double FindDoubleProperty(const map<string, string>& rProperties, const string& key, double defaultValue, bool caseSensitive = false);

    /**
     * @return the index of the given object or -1 if the object is
     * not found. operator== must be defined for the type T.
     **/
    template<typename T> static int IndexOf(vector<T> const & rVector, T const & rElement) {
        for (long i = 0; i < rVector.size(); i++) {
            if (rVector[i] == rElement) {
                return i;
            }
        }
        return -1;
    }

    template<typename T> static bool Contains(vector<T> const & rVector, T const & rElement) {
        return IndexOf(rVector, rElement) >= 0;
    }

};

} /* namespace utils */

#endif /* UTILS_H_ */
