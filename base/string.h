#include <string>
#include <ostream>
#include "linkedlist.h"
#include "arraylist.h"

#ifndef STRING_H
#define STRING_H

namespace base {

	
class String : public std::string
{
 	public:
        static const char CHAR_TAB = '\t';
        static const char CHAR_SPACE = ' ';
 	public:
        String() {};
        
        /**
         * Constructs a string from another string
         **/
        String(const String& s) : std::string(s) {};

    private:
        /**
         * Constructs a string from std::string
         **/
        String(std::string s) : std::string(s) {};

    public:
        /**
         * Constructs a string from a regular C string
         **/
        String(const char* s) : std::string(s) {};

        /**
         * Constructs a string from int value
         **/
        String(int num);

        /**
         * Constructs a string from long value
         **/
        String(long num);

        /**
         * Constructs a string from double value
         **/
        String(double num);

        /**
         * Converts this string to double
         **/
        double ToDouble() const;
        
        /**
         * Converts this string to int
         **/
        int ToInt() const;
        
        /**
         * Converts this string to long
         **/
        long ToLong() const;
        
        /**
         * This function returns a const pointer to a regular 
         * C string, identical to the current string. The returned 
         * string is null-terminated. 
         *
         * Note that since the returned pointer is of type 
         * (C/C++ Keywords) const, the character data that 
         * ToChar returns cannot be modified.
         **/
        const char* GetChars() const {
            return c_str();
        }

        /**
         * Returns the length of this string
         **/
        int Length() const {
            return length();
        }
        
        /**
         * Clears this string
         **/
        void Clear() {
            return clear();
        }
        
        /**
         * Removes all leading and trailing whitespaces from this string
         * @return reference to this string
         **/
        String& Trim();
        
        /**
         * Searches for a substring in this string from the
         * given start index.
         * @param rStr substring to find
         * @param startIndex start index or 0 if omitted
         * @return index of the first character of the substring 
         * or -1 if substring is not found
         **/
        int Find(const String& rStr, int startIndex = 0) const;

        /**
         * Searches for all occurances of the specified substring
         * starting from the given start index.
         * @param rIndices the found indices are inserted here
         * @param rStr substring to find
         * @param startIndex start index or 0 if omitted
         **/
         void FindAll(List<int>& rIndices, const String& rStr, int startIndex = 0) const;
        
        /**
         * Returns the substring starting from the start index and 
         * ending at the end index. The character at end index is not
         * included.
         * @param from start index or 0 if omitted
         * @param to end infex, if omitted or -1 the substring from
         * start index to the end of the string is returned
         **/
        String SubString(int from = 0, int to = -1) const;
        
        /**
         * Finds next whitespace character starting from the
         * given position
         * @return index of the whitespace character or -1 if
         * not found
         **/
        int FindWhitespace(int startIndex = 0) const;
        
        /**
         * Splits the string to an array using the given separator
         * @param separator separator
         * @param trim specifies whether to trim the chunks after splitting
         * @return list of chunks
         **/
        ArrayList<String> Split(const String& separator, bool trim = true) const;
        
        /**
         * Returns the property with the given name cointained in the
         * string. Properties are defined in the string as "name = value"
         * pairs. The value may be put between quotation marks to allow 
         * spaces.
         * If no property is found empty string is returned
         **/
        String GetProperty(const String& rPropertyName) const;

        //--------------------------------------------
        // Member operators
        
        String& operator=(const char* str) {
            assign(str);
            return *this;
        }
        
        String& operator=(const std::string& str) {
            assign(str);
            return *this;
        }
        
        String& operator=(const String& str) {
            assign(str);
            return *this;
        }

//        String& operator=(int num) {
//            assign(String(num));
//            return *this;
//        }
//
//        String& operator=(double num) {
//            assign(String(num));
//            return *this;
//        }
        
        bool operator==(const char* str) const {
            return compare(str) == 0;
        }
        
        bool operator==(const String& str) const {
            return compare(str) == 0;
        }
        
        char operator[](int index) const {
            return at(index);
        }
    
        String operator+(const String& s) const {
            return String(*this).append(s);
        }
        
        String operator+(char c) const {
            return *this + String(c);
        }
        
        String operator+(const char* c) const {
            return *this + String(c);
        }
        
        String operator+(int num) const {
            return *this + String(num);
        }

        String operator+(long num) const {
            return *this + String(num);
        }
        
        String operator+(double num) const {
            return *this + String(num);
        }

        friend std::ostream& operator<<(std::ostream& o, const String& s) {
            return o << s.GetChars();
        }
        
        
    public:
        static bool IsLetterOrDigit(char ch);
        static bool IsLetter(char ch);
        static bool IsDigit(char ch);
        static bool IsWhitespace(char ch);
};
}


#endif
