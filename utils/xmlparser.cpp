#include "xmlparser.h"
#include <cassert>

#include <iostream>

using namespace utils;

XmlParser::XmlParser(const string& rFileName) {
	if (rFileName.length() > 0) {
		Load(rFileName);
	}
}

XmlParser::~XmlParser() {
}

XmlParser::XmlElement* XmlParser::Load(const string& rFileName) {
	if (rFileName.length() > 0) {
		XmlElement* p_element = new XmlElement("root");
		try {
			TextReader reader(rFileName);
			p_element->Parse(reader);
		}
		catch(const std::exception& e) {
			cout << "Exception caught: " << e.what() << endl;
		}

	    return p_element;
	}
    return nullptr;
}

//----------------------------------------------
// XmlElement

XmlParser::XmlElement::XmlElement(const string& rName, XmlElement* pParent) :
    mName(rName),
	mpParent(pParent)
{
	if (mpParent) {
		mDepth = mpParent->mDepth + 1;
	} else {
		mDepth = 0;
	}
    Object::Dbg(string("------ XmlElement: ") + mName);
    
}

XmlParser::XmlElement::~XmlElement()
{
    for (auto i = mSubElements.begin(); i != mSubElements.end(); i++) {
        delete (*i);
    }
    mSubElements.clear();
}

void XmlParser::XmlElement::Parse(TextReader& rReader)
{
	try {
		while (rReader.get_depth() == mDepth + 1) {
			rReader.read();
			switch (rReader.get_node_type()) {
				case TextReader::Element: {
					AddSubElement(new XmlElement(rReader.get_name()));
					if (rReader.has_attributes()) {
						rReader.move_to_first_attribute();
						do {
							mSubElements.back()->AddAttribute(rReader.get_name(), rReader.get_value());
						} while (rReader.move_to_next_attribute());
						rReader.move_to_element();
					}
					break;
				}
				case TextReader::Text:
					mInnerText = rReader.read_string();
					break;
				default:
					break;
			}
		}
		if (rReader.get_depth() > mDepth + 1) {
			mSubElements.back()->Parse(rReader);
		}
	} catch(const std::exception& e) {
		cout << "Exception caught: " << e.what() << endl;
	}
}    

string XmlParser::XmlElement::ToString() const
{
	string output = string("    ", mDepth) + "<" + mName;
	if (mSubElements.size() == 0) {
		output = output + "/>\n";
	}
	else {
		output = output + ">\n";
	    for (auto i = mSubElements.begin(); i != mSubElements.end(); i++) {
			output = output + string("    ", (*i)->mDepth) + (*i)->ToString();
		}
		output = output + string("    ", mDepth) + "</" + mName + ">\n";
	}
	return output;
}

void XmlParser::XmlElement::Save(const string& rFileName) const {
	ofstream out(rFileName);
	out << ToString();
	out.close();
}
