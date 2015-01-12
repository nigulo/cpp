#include "xmlparser.h" // class's header file

using namespace utils;
using namespace base;

// class constructor
XmlParser::XmlParser(const String& rFileName) 
{
	mpData = 0;
	if (rFileName.Length() > 0) {
		Load(rFileName);
	}
}

// class destructor
XmlParser::~XmlParser()
{
	if (mpData) {
		delete mpData;
	}
}

XmlParser::XmlElement* XmlParser::Load(const String& rFileName) 
{
	if (!mpData || rFileName.Length() > 0) {
	    String data;
		char buf[256];
	    //--------------------------------
	    // Load scene data
	    std::ifstream in; 
	    Debug(rFileName.GetChars());
	    in.open(rFileName.GetChars());
	    assert(in);
		while (in.getline(buf, sizeof(buf))) {
			data = data + buf + "\n";
		}
		in.close();
		mpData = new XmlElement("root", "root", "", data, 0, data.Length());
		mpData->Parse();
	    data.Clear();
	}
    return mpData;
}

//----------------------------------------------
// XmlElement

XmlParser::XmlElement::XmlElement(const String& rType)
{
	mType = rType;
	mParams = rType;
}

XmlParser::XmlElement::XmlElement(const String& rType, const String& rName, const String& rParams, String& rData, int startIndex, int endIndex) :
    mType(rType), 
    mName(rName), 
    mParams(rParams),
    mStartIndex(startIndex), 
    mEndIndex(endIndex)
{
    mpData = &rData;
    Object::Dbg(String("------ XmlElement: ") + mType + String(", ") + mName);// + String(", ") + mStartIndex + String(", ") + mEndIndex);
    
}

XmlParser::XmlElement::~XmlElement()
{
    for (LinkedList<XmlElement*>::Iterator i = mSubElements.Begin(); !i.Done(); i++) {
        delete (*i);
    }
    mSubElements.Clear();
}

void XmlParser::XmlElement::AddSubElement(XmlElement* pElement) {
	mSubElements.Add(pElement);
}

void XmlParser::XmlElement::SetData(const String& rData) {
	mData = rData; 
}

void XmlParser::XmlElement::AddParameter(const String& rName, const String& rValue) {
	if (mParams.Length() > 0) {
		mParams = mParams + " ";
	}
	mParams = mParams + rName + (rValue.Length() > 0 ? (String("=\"") + rValue + "\"") : ""); 
}

void XmlParser::XmlElement::Parse() 
{
    int startIndex = mStartIndex;
    bool leaf = true;
    while (startIndex < mEndIndex) {
        Object::Dbg(String("Parse ") + (double) startIndex);
        
        XmlElement* pElement = FindNextSubElement(&startIndex);
        if (pElement) {
            leaf = false;
            mSubElements.Add(pElement);
            pElement->Parse();
        }
        else {
            break;
        }
    }
    if (leaf) {
        mData = mpData->SubString(startIndex, mEndIndex);
        Object::Dbg((String) "Leaf element data _" + mData + "_");
    }
}    

XmlParser::XmlElement* XmlParser::XmlElement::FindNextSubElement(int* pStartIndex)
{
    int start;
    while (true) {
        start = mpData->Find("<", *pStartIndex);
        Object::Dbg(String("FindNextSubElement start: ") + start);
        
        if (start < 0 || start >= mEndIndex) {
            // No XML tags found, return 0
            Object::Dbg((String) "FindNextSubElement substr: _" + mpData->SubString(*pStartIndex, mEndIndex) + "_");
            return 0;
        }
        else if (mpData->SubString(start + 1, start + 4) == "!--") {
            // Comment tag found
            int end = start;
            do {
                start = mpData->Find("<!--", start + 1);
                end = mpData->Find("-->", end + 1);
            } while (start >= 0 && start < end);
            assert(end >= 0);
            *pStartIndex = end + ((String) "-->").Length();
            // This is a comment tag, return 0
            Object::Dbg("Comment found");
            continue;
        }
        break;
    }

    int end = mpData->Find(">", start);
    Object::Dbg((String) "FindNextSubElement end: " + end);
    
    assert(end < mEndIndex);
    String tag = mpData->SubString(start + 1, end);
    tag.Trim();
    bool endTag = true;
    if (tag[tag.Length() - 1] == '/') {
        Object::Dbg("Element with no end tag");
        tag = tag.SubString(0, tag.Length() - 1);
        // The tag is self-closing, no end tag is present
        endTag = false;
    }
    Object::Dbg((String) "FindNextSubElement tag: _" + tag + "_");
    
    tag.Trim();
    Object::Dbg((String) "FindNextSubElement tag trimmed: _" + tag + "_");
    
    int whiteIndex = tag.FindWhitespace();
    Object::Dbg((String) "FindNextSubElement whiteIndex: " + whiteIndex);
    
    String type = tag.SubString(0, whiteIndex);
    Object::Dbg((String) "FindNextSubElement type: " + type);
    
    String name = tag.GetProperty("name");
    Object::Dbg((String) "FindNextSubElement name: " + name);
    
    int elementEnd = end + 1;
    *pStartIndex = elementEnd;
    if (endTag) {
        // The end tag is present
        ArrayList<int> start_indices(100);
        mpData->FindAll(start_indices, (String) "<" + type, end);
        Object::Dbg(String("siiin"));
        ArrayList<int> end_indices(100);
        mpData->FindAll(end_indices, (String) "</" + type, end);
        ArrayList<int> comment_starts(100);
        mpData->FindAll(comment_starts, (String) "<!--", end);
        ArrayList<int> comment_ends(100);
        mpData->FindAll(comment_ends, (String) "-->", end);
        
        int num_tags = 1;
        int num_comments = 0;
        int i_start = 0;
        int i_end = 0;
        int i_comm_start = 0;
        int i_comm_end = 0;
        Object::Dbg(String("start size: ") + start_indices.Size());
        Object::Dbg(String("end size: ") + end_indices.Size());
        do {
            bool start = false;
            bool comment = false;
            int index;
            assert(i_end < end_indices.Size());
            if (i_start < start_indices.Size() && start_indices[i_start] < end_indices[i_end]) {
                start = true;
                index = start_indices[i_start];
            }
            else {
                index = end_indices[i_end];
            }
            if (i_comm_start < comment_starts.Size()) {
                if (comment_starts[i_comm_start] < index) {
                    index = comment_starts[i_comm_start];
                    comment = true;
                }
            }
            if (i_comm_end < comment_ends.Size()) {
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
            Object::Dbg(String("num_tags: ") + num_tags);
            Object::Dbg(String("i_start: ") + i_start);
            Object::Dbg(String("i_end: ") + i_end);
        } while (num_tags > 0);
        elementEnd = end_indices[i_end - 1];
        
        //elementEnd = mpData->Find((String) "</" + type, end);
        *pStartIndex = elementEnd + ((String) "</" + type + ">").Length();
    }
    Object::Dbg((String) "FindNextSubElement elementEnd: " + elementEnd);
    
    assert(elementEnd < mEndIndex);
    return new XmlElement(type, name, tag, *mpData, end + 1, elementEnd);
}

String XmlParser::XmlElement::ToString() const
{
	return ToString("");
}

String XmlParser::XmlElement::ToString(const String& indent) const 
{
	
	String output = indent + "<" + mParams;
	if (mSubElements.Size() == 0 && mData.Length() == 0) {
		output = output + "/>\n";
	}
	else {
		output = output + ">\n";
		for (int i = 0; i < mSubElements.Size(); i++) {
			output = output + mSubElements[i]->ToString(indent + "    ");
		}
		if (mData.Length() > 0) {
			output = output + indent + mData + "\n";
		}
		output = output + indent + "</" + mType + ">\n";
	}
	return output;
}

void XmlParser::XmlElement::Save(const String& rFileName) const {
	std::ofstream out(rFileName.GetChars());
	out << ToString();
	out.close();
}
