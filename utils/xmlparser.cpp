#include "xmlparser.h"
#include <cassert>

using namespace utils;
using namespace base;

XmlParser::XmlParser(const string& rFileName) {
	mpData = nullptr;
	if (rFileName.length() > 0) {
		Load(rFileName);
	}
}

XmlParser::~XmlParser() {
	if (mpData) {
		delete mpData;
	}
}

XmlParser::XmlElement* XmlParser::Load(const string& rFileName) {
	if (!mpData || rFileName.length() > 0) {
	    string data;
		char buf[256];
	    //--------------------------------
	    // Load scene data
	    ifstream in;
	    Debug(rFileName);
	    in.open(rFileName);
	    assert(in);
		while (in.getline(buf, sizeof(buf))) {
			data = data + buf + "\n";
		}
		in.close();
		mpData = new XmlElement("root", "root", "", data, 0, data.length());
		mpData->Parse();
	    data.clear();
	}
    return mpData;
}

//----------------------------------------------
// XmlElement

XmlParser::XmlElement::XmlElement(const string& rType, const string& rName, const string& rParams, string& rData, int startIndex, int endIndex) :
    mType(rType), 
    mName(rName), 
    mParams(rParams),
    mStartIndex(startIndex), 
    mEndIndex(endIndex)
{
    mpData = &rData;
    Object::Dbg(string("------ XmlElement: ") + mType + string(", ") + mName);// + string(", ") + mStartIndex + string(", ") + mEndIndex);
    
}

XmlParser::XmlElement::~XmlElement()
{
    for (auto i = mSubElements.begin(); i != mSubElements.end(); i++) {
        delete (*i);
    }
    mSubElements.clear();
}

void XmlParser::XmlElement::AddSubElement(XmlElement* pElement) {
	mSubElements.push_back(pElement);
}

void XmlParser::XmlElement::SetData(const string& rData) {
	mData = rData; 
}

void XmlParser::XmlElement::AddParameter(const string& rName, const string& rValue) {
	if (mParams.length() > 0) {
		mParams = mParams + " ";
	}
	mParams = mParams + rName + (rValue.length() > 0 ? (string("=\"") + rValue + "\"") : "");
}

void XmlParser::XmlElement::Parse() 
{
    int startIndex = mStartIndex;
    bool leaf = true;
    while (startIndex < mEndIndex) {
        Object::Dbg(string("Parse ") + to_string(startIndex));
        
        XmlElement* pElement = FindNextSubElement(&startIndex);
        if (pElement) {
            leaf = false;
            mSubElements.push_back(pElement);
            pElement->Parse();
        }
        else {
            break;
        }
    }
    if (leaf) {
        mData = mpData->substr(startIndex, mEndIndex - startIndex);
        Object::Dbg((string) "Leaf element data _" + mData + "_");
    }
}    

XmlParser::XmlElement* XmlParser::XmlElement::FindNextSubElement(int* pStartIndex)
{
    int start;
    while (true) {
        start = Utils::Find(*mpData, "<", *pStartIndex);
        Object::Dbg(string("FindNextSubElement start: ") + to_string(start));
        
        if (start < 0 || start >= mEndIndex) {
            // No XML tags found, return 0
            Object::Dbg((string) "FindNextSubElement substr: _" + mpData->substr(*pStartIndex, mEndIndex - *pStartIndex) + "_");
            return 0;
        }
        else if (mpData->substr(start + 1, 3) == "!--") {
            // Comment tag found
            int end = start;
            do {
                start = Utils::Find(*mpData, "<!--", start + 1);
                end = Utils::Find(*mpData, "-->", end + 1);
            } while (start >= 0 && start < end);
            assert(end >= 0);
            *pStartIndex = end + ((string) "-->").length();
            // This is a comment tag, return 0
            Object::Dbg("Comment found");
            continue;
        }
        break;
    }

    int end = Utils::Find(*mpData, ">", start);
    Object::Dbg((string) "FindNextSubElement end: " + to_string(end));
    
    assert(end < mEndIndex);
    string tag = mpData->substr(start + 1, end - start - 1);
    Utils::Trim(tag);
    bool endTag = true;
    if (tag[tag.length() - 1] == '/') {
        Object::Dbg("Element with no end tag");
        tag = tag.substr(0, tag.length() - 1);
        // The tag is self-closing, no end tag is present
        endTag = false;
    }
    Object::Dbg((string) "FindNextSubElement tag: _" + tag + "_");
    
    Utils::Trim(tag);
    Object::Dbg((string) "FindNextSubElement tag trimmed: _" + tag + "_");
    
    int whiteIndex = Utils::FindWhitespace(tag);
    Object::Dbg((string) "FindNextSubElement whiteIndex: " + to_string(whiteIndex));
    
    string type = tag.substr(0, whiteIndex);
    Object::Dbg((string) "FindNextSubElement type: " + type);
    
    string name = Utils::GetProperty(tag, "name");
    Object::Dbg((string) "FindNextSubElement name: " + name);
    
    int elementEnd = end + 1;
    *pStartIndex = elementEnd;
    if (endTag) {
        // The end tag is present
        vector<int> start_indices(100);
        Utils::FindAll(*mpData, start_indices, (string) "<" + type, end);
        Object::Dbg(string("siiin"));
        vector<int> end_indices(100);
        Utils::FindAll(*mpData, end_indices, (string) "</" + type, end);
        vector<int> comment_starts(100);
        Utils::FindAll(*mpData, comment_starts, (string) "<!--", end);
        vector<int> comment_ends(100);
        Utils::FindAll(*mpData, comment_ends, (string) "-->", end);
        
        int num_tags = 1;
        int num_comments = 0;
        int i_start = 0;
        int i_end = 0;
        int i_comm_start = 0;
        int i_comm_end = 0;
        Object::Dbg(string("start size: ") + to_string(start_indices.size()));
        Object::Dbg(string("end size: ") + to_string(end_indices.size()));
        do {
            bool start = false;
            bool comment = false;
            int index;
            assert(i_end < end_indices.size());
            if (i_start < start_indices.size() && start_indices[i_start] < end_indices[i_end]) {
                start = true;
                index = start_indices[i_start];
            }
            else {
                index = end_indices[i_end];
            }
            if (i_comm_start < comment_starts.size()) {
                if (comment_starts[i_comm_start] < index) {
                    index = comment_starts[i_comm_start];
                    comment = true;
                }
            }
            if (i_comm_end < comment_ends.size()) {
                if (comment_ends[i_comm_end] < index) {
                    index = comment_ends[i_comm_end];
                    start = false;
                    comment = true;
                }
            }
            if (comment) {
                if (start) {
                    num_comments++;
                    i_comm_start++;
                }
                else {
                    num_comments--;
                    i_comm_end++;
                }
            }
            else {
                if (start) {
                    i_start++;
                    if (num_comments <= 0) {
                        num_tags++;
                    }
                }
                else {
                    i_end++;
                    if (num_comments <= 0) {
                        num_tags--;
                    }
                }
            }
            
            // If the next assert fails, the end tag is not found
            //assert(i_end < end_indices.Size());
            //if (i_start < start_indices.Size() && start_indices[i_start] < end_indices[i_end]) {
            //    num_tags++;
            //    i_start++;
            //}
            //else {
            //    num_tags--;
            //    i_end++;
            //}
            Object::Dbg(string("num_tags: ") + to_string(num_tags));
            Object::Dbg(string("i_start: ") + to_string(i_start));
            Object::Dbg(string("i_end: ") + to_string(i_end));
        } while (num_tags > 0);
        elementEnd = end_indices[i_end - 1];
        
        //elementEnd = mpData->Find((string) "</" + type, end);
        *pStartIndex = elementEnd + ((string) "</" + type + ">").length();
    }
    Object::Dbg((string) "FindNextSubElement elementEnd: " + to_string(elementEnd));
    
    assert(elementEnd < mEndIndex);
    return new XmlElement(type, name, tag, *mpData, end + 1, elementEnd);
}

string XmlParser::XmlElement::ToString() const
{
	return ToString("");
}

string XmlParser::XmlElement::ToString(const string& indent) const
{
	
	string output = indent + "<" + mParams;
	if (mSubElements.size() == 0 && mData.length() == 0) {
		output = output + "/>\n";
	}
	else {
		output = output + ">\n";
	    for (auto i = mSubElements.begin(); i != mSubElements.end(); i++) {
			output = output + (*i)->ToString(indent + "    ");
		}
		if (mData.length() > 0) {
			output = output + indent + mData + "\n";
		}
		output = output + indent + "</" + mType + ">\n";
	}
	return output;
}

void XmlParser::XmlElement::Save(const string& rFileName) const {
	ofstream out(rFileName);
	out << ToString();
	out.close();
}
