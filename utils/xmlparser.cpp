#include "xmlparser.h"
#include <cassert>

#include <iostream>
#include <libxml++/parsers/textreader.h>

using namespace xmlpp;

using namespace utils;

XmlParser::XmlParser(const string& rFileName) : mFileName(rFileName) {
}

XmlParser::~XmlParser() {
}

void Parse(TextReader& rReader, XmlParser::XmlElement* p_elem, int depth)
{
	try {
		while (rReader.get_depth() > depth) {
			//cout << string(4 * rReader.get_depth(), ' ') << rReader.get_name() << endl;
			if (rReader.get_depth() > depth + 1) {
				Parse(rReader, p_elem->GetSubElements().back(), depth + 1);
			}
			switch (rReader.get_node_type()) {
				case TextReader::NodeType::Element: {
					p_elem->AddSubElement(new XmlParser::XmlElement(rReader.get_name(), p_elem));
					if (rReader.has_attributes()) {
						rReader.move_to_first_attribute();
						do {
							p_elem->GetSubElements().back()->AddAttribute(rReader.get_name(), rReader.get_value());
						} while (rReader.move_to_next_attribute());
						rReader.move_to_element();
					}
					break;
				}
				case TextReader::NodeType::Text:
					p_elem->SetInnerText(rReader.read_string());
					break;
				default:
					break;
			}
			if (!rReader.read()) {
				cout << "Premature end of xml " << endl;
				break;
			}
		}
	} catch(const std::exception& e) {
		cout << "Exception caught: " << e.what() << endl;
	}
}

XmlParser::XmlElement* XmlParser::Load() {
	if (mFileName.length() > 0) {
		XmlElement* p_element = nullptr;
		try {
			TextReader reader(mFileName);
			reader.read();
			p_element = new XmlElement(reader.get_name());
			reader.read();
			Parse(reader, p_element, 0);
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
	mpParent(pParent),
	mDepth(pParent ? mpParent->mDepth + 1 : 0)
{
}

XmlParser::XmlElement::~XmlElement()
{
    for (auto i = mSubElements.begin(); i != mSubElements.end(); i++) {
        delete (*i);
    }
    mSubElements.clear();
}
/*
void XmlParser::XmlElement::Parse(TextReader& rReader)
{
	try {
		while (rReader.get_depth() > mDepth) {
			//cout << string(4 * rReader.get_depth(), ' ') << rReader.get_name() << endl;
			if (rReader.get_depth() > mDepth + 1) {
				mSubElements.back()->Parse(rReader);
			}
			switch (rReader.get_node_type()) {
				case TextReader::Element: {
					AddSubElement(new XmlElement(rReader.get_name(), this));
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
			if (!rReader.read()) {
				cout << "Premature end of xml " << endl;
				break;
			}
		}
	} catch(const std::exception& e) {
		cout << "Exception caught: " << e.what() << endl;
	}
}    
*/

string XmlParser::XmlElement::ToString() const
{
	string output = string(4 * mDepth, ' ') + "<" + mName;
	for (auto&& attr : mAttributes) {
		output = output + " " + attr.first + "=\"" + attr.second + "\"";
	}
	if (mInnerText.length() == 0 && mSubElements.size() == 0) {
		output = output + "/>\n";
	}
	else {
		output = output + ">\n";
	    for (auto i = mSubElements.begin(); i != mSubElements.end(); i++) {
			output = output + (*i)->ToString();
		}
		output = output + mInnerText + "\n";
		output = output + string(4 * mDepth, ' ') + "</" + mName + ">\n";
	}
	return output;
}

void XmlParser::XmlElement::Save(const string& rFileName) const {
	ofstream out(rFileName);
	out << ToString();
	out.close();
}
