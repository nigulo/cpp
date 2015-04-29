#ifndef XMLPARSER_H
#define XMLPARSER_H

#include "base/object.h"
#include "utils.h"
#include <list>

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
        public:
            XmlElement(const string& rType, const string& rName, const string& rParams, string& data, int startIndex, int endIndex);
            
            void AddParameter(const string& rName, const string& rValue = "");
            void AddSubElement(XmlElement* pElement);
            void SetData(const string& rData);
            
            string GetParameter(const string& rParamName) const {
            	return Utils::GetProperty(mParams, rParamName);
            }
            
            list<XmlElement*>& GetSubElements() {
            	return mSubElements;
            }
            
            virtual ~XmlElement();
            void Parse();
            string ToString() const;

    		void Save(const string& rFileName) const;
        private:
            XmlElement* FindNextSubElement(int* pStartIndex);
            string ToString(const string& indent) const;
        public:
            string mType;
            string mName;
            string mParams;
            list<XmlElement*> mSubElements;
            string mData;
        private:
            int mStartIndex;
            int mEndIndex;
            string* mpData;
    };
    
	public:
		// class constructor
		XmlParser(const string& rFileName = "");
		XmlElement* Load(const string& rFileName = "");
		// class destructor
		virtual ~XmlParser();
	private:
        XmlElement* mpData;
};
}
#endif // XMLPARSER_H
