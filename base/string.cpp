#include <stdlib.h>
#include "string.h"
#include "object.h"
#include "arraylist.h"

using namespace std;
using namespace base;

String::String(int num) : string(to_string(num)) {
}

String::String(long num) : string(to_string(num)) {
}

String::String(double num) : string(to_string(num)) {
}

double String::ToDouble() const
{
	return stod(*this);
}

int String::ToInt() const
{
	return stoi(*this);
}

long String::ToLong() const
{
	return stol(*this);
}

int String::Find(const String& rStr, int startIndex) const
{
    assert(startIndex >= 0 && startIndex < Length());
    return find(rStr, startIndex);
}

void String::FindAll(List<int>& rIndices, const String& rStr, int startIndex) const
{
    int index = startIndex - 1;
    while ((index = Find(rStr, index + 1)) >= 0) {
        Object::Dbg(to_string(index));
        rIndices.Add(index);
        //p_indices->Add(index);
        Object::Dbg(to_string(-1));
    }
    //return p_indices;
}
String String::SubString(int from, int to) const
{
    assert(from >= 0 && from < Length());
    if (to < 0) {
        to = Length();
    }
    assert(to >= from && to <= Length());
    return substr(from, to - from);
}

String& String::Trim()
{
    int startIndex = 0;
    while (startIndex < length() && IsWhitespace((*this)[startIndex])) {
        startIndex++;
    }
    int endIndex = length() - 1;
    while (endIndex > 0 && IsWhitespace((*this)[endIndex])) {
        endIndex--;
    }
    if (endIndex < startIndex) {
        clear();
    }
    else {
        assign(substr(startIndex, endIndex - startIndex + 1));
    }
    return *this;
}

bool String::IsLetter(char ch)
{
    if (ch >= 65 && ch <= 80 || ch == '_' || ch >= 97 && ch <= 122) {
        return true;
    }
    return false;
}

bool String::IsDigit(char ch)
{
    if (ch >= 48 && ch <= 57) {
        return true;
    }
    return false;
}

bool String::IsLetterOrDigit(char ch)
{
    return (IsLetter(ch) | IsDigit(ch));
}

bool String::IsWhitespace(char ch)
{
    if (ch == CHAR_TAB || ch == CHAR_SPACE) {
        return true;
    }
    return false;
}

int String::FindWhitespace(int startIndex) const
{
    int index1 = find(CHAR_TAB, startIndex);
    int index2 = find(CHAR_SPACE, startIndex);
    if (index1 >= 0 && index1 < index2) {
        return index1;
    }
    return index2;
}

ArrayList<String> String::Split(const String& separator, bool trim) const 
{
    //Object::Dbg(String("Split 0"));
    int index = -1;
    int count = 1;
    while (true) {
        //Object::Dbg(String("Split 1"));
        index = find(separator, index + 1);
        //Object::Dbg(String("Split 1: ") + index);
        if (index < 0) {
            break;
        }
        count++;
    }
    ArrayList<String> ret_val(count);
    //String* ret_val = new String[count];
    index = 0;
    //Object::Dbg(String("Split 2"));
    for(int i = 0; i < count; i++) {
        //Object::Dbg(String("Split 3"));
        int new_index = find(separator, index);
        //Object::Dbg(String("Split 3: ") + new_index);
        String s;
        if (new_index < 0) {
            s = substr(index);
        }
        else {
            s = substr(index, new_index - index);
        }
        //Object::Dbg(String("SplitA ") + i + ": " + s);
        if (trim) {
            s.Trim();
        }
        //Object::Dbg(String("SplitB ") + i + ": " + s);
        ret_val.Add(s);
        index = new_index + 1;
    }
    return ret_val;
}

String String::GetProperty(const String& rPropertyName) const
{
    int index = find(rPropertyName);
    //Object::Dbg((String) "GetProperty index: " + index);
    
    if (index >= length()) {
        return String();
    }
    index += rPropertyName.length();
    //Object::Dbg((String) "GetProperty index_new: " + index);
    if (index >= length()) {
        return String();
    }
    String str = substr(index);
    str.Trim();
    //Object::Dbg((String) "GetProperty str: " + str);
    
    if (str[0] != '=') {
        return String();
    }
    str = str.substr(1);
    str.Trim();
    //Object::Dbg((String) "GetProperty str: " + str);
    
    if (str[0] == '"') {
        str = str.substr(1);
        //Object::Dbg((String) "GetProperty str1: " + str);
        
        index = str.find('"');
        //Object::Dbg((String) "GetProperty index1: " + index);
        
        assert(index < str.length());
    }
    else if ((*this)[0] == '\'') {
        str = str.substr(1);
        //Object::Dbg((String) "GetProperty str2: " + str);
        
        index = str.find('\'');
        //Object::Dbg((String) "GetProperty index2: " + index);
        
        assert(index < str.length());
    }
    else {
        index = str.FindWhitespace(0);
        //Object::Dbg((String) "GetProperty index3: " + index);
        
    }
    return str.substr(0, index);
}
