#ifndef XMLPARSER_H
#define XMLPARSER_H

#include "base/object.h"
#include "utils.h"
#include <list>
#include <map>

using namespace base;
using namespace std;

namespace utils {

/**
 * A primitive XML parser
 */
class XmlParser : public Object
{

    public:
    class XmlElement
    {
    	friend class XmlParser;
        public:
            XmlElement(const string& rName, XmlElement* pParent = nullptr);

            const string& GetName() const {
            	return mName;
            }
            
            const string GetAttribute(const string& rName) const {
            	auto i = mAttributes.find(rName);
            	if (i != mAttributes.end()) {
            		return i->second;
            	}
            	return "";
            }

            const string& GetInnerText() const {
            	return mInnerText;
            }

            list<XmlElement*>& GetSubElements() {
            	return mSubElements;
            }
            
            virtual ~XmlElement();
            string ToString() const;

    		void Save(const string& rFileName) const;
            void SetInnerText(const string& rInnerText) {
            	mInnerText = rInnerText;
            }
            void AddAttribute(const string& name, const string& value) {
            	mAttributes.insert({name, value});
            }
            void AddSubElement(XmlElement* pElement) {
            	mSubElements.push_back(pElement);
            }
        private:
            string mName;
            XmlElement* mpParent;
            int mDepth;
            map<string, string> mAttributes;
            string mInnerText;
            list<XmlElement*> mSubElements;
    };
    
	public:
		XmlParser(const string& rFileName);
		XmlElement* Load();
		virtual ~XmlParser();
	private:
		const string mFileName;
};
}
#endif // XMLPARSER_H
