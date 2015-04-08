#ifndef XMLPARSER_H
#define XMLPARSER_H

#include "base/object.h"
#include "base/string.h"
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
            XmlElement(const String& rType, const String& rName, const String& rParams, String& data, int startIndex, int endIndex);
            
            void AddParameter(const String& rName, const String& rValue = "");
            void AddSubElement(XmlElement* pElement);
            void SetData(const String& rData);
            
            String GetParameter(const String& rParamName) const {
            	return mParams.GetProperty(rParamName);
            }
            
            list<XmlElement*>& GetSubElements() {
            	return mSubElements;
            }
            
            virtual ~XmlElement();
            void Parse();
            String ToString() const;

    		void Save(const String& rFileName) const;
        private:
            XmlElement* FindNextSubElement(int* pStartIndex);
            String ToString(const String& indent) const;
        public:
            String mType;
            String mName;
            String mParams;
            list<XmlElement*> mSubElements;
            String mData;
        private:
            int mStartIndex;
            int mEndIndex;
            String* mpData;
    };
    
	public:
		// class constructor
		XmlParser(const String& rFileName = "");
		XmlElement* Load(const String& rFileName = "");
		// class destructor
		virtual ~XmlParser();
	private:
        XmlElement* mpData;
};
}
#endif // XMLPARSER_H
