#include <time.h>
#include "object.h"
using namespace std;
using namespace base;


ofstream Object::msOut("log.txt");

//#ifndef DEBUG
//bool Object::msDebug = false;
//#else
bool Object::msDebug = true;
//#endif

void Object::Flush()
{
	msOut.flush();
}

Object::Object() : mName("")
{
}

Object::Object(const string& rName) : mName(rName)
{
}

Object::Object(const Object& rObj) : mName(rObj.mName)
{
}

Object* Object::Clone() const
{
    Debug("Object.Clone");
    return new Object(*this);
}

void Object::Copy(const Object& rObj) 
{
    mName = rObj.mName;
}

// class destructor
Object::~Object()
{
}

void Object::Debug(const string& rText) const
{
    if (msDebug) {
        msOut << GetMillis() << " " << mName << " " << rText << "\n";
        msOut.flush();
    }
}

void Object::Dbg(const string& rText)
{
    if (msDebug && msOut) {
        msOut << GetMillis() << " " << rText << "\n";
        msOut.flush();
    }
}

long Object::GetMillis()
{
    return clock();
}

string Object::ToString() const
{
    return mName;
}
