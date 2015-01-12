#ifndef OBJECT_H
#define OBJECT_H

#include <fstream>

#include "string.h"

//#ifdef DEBUG
//#define D(text) out << GetName() << " " << text << "\n"
//#else
//#define D(text) 
//#endif

using namespace std;

namespace base {
    /*
     * Base class of all objects
     */
    class Object
    {
    	public:
            /**
    		 * Default constructor. Creates an object with an empty name.
    		 **/
    		Object();
    		
    		/**
    		 * Copy constructor
    		 **/
    		Object(const Object& rObj);
    		
    		/**
    		 * Constructs an object with the given name
    		 **/
    		Object(const String& rName);

            /**
             * Class destructor
             **/           
    		virtual ~Object();

    		/**
    		 * Clones this object. Override this method in sub-classes
    		 * to get deep copies of the objects. Use the following
    		 * technique: create a new object in Clone method and call
    		 * method Copy with *this as argument.
    		 * @return the copy of this object
    		 * @see void Copy(const Object& rObj)
    		 **/
    		virtual Object* Clone() const;
    		
    		/**
    		 * Copies the fields of the given object into this
             * object. It must call the superclass 
             * version of the Copy to make a deep copy.
    		 * @param rObj object to copy into this object
    		 * @see Object* Clone() const
    		 **/
    		virtual void Copy(const Object& rObj);
    		
            /**
             * @return the name of this object
             **/
    		const String& Name() const {
            	return mName;
            }

            /**
             * Prints the given text prefixed with the
             * object's name to output stream if the debugging
             * is enabled.
             **/            
    		virtual void Debug(const String& rText) const;

            /**
             * Prints the given text to output stream if 
             * the debugging is enabled.
             **/            
            static void Dbg(const String& rText);
            
            /**
             * Flushes the output stream
             **/
            static void Flush();

            /**
             * @return the system milliseconds
             **/
            static long GetMillis();

            /**
             * Override this method to get special results
             * @return the string representation of this object
             **/
            virtual String ToString() const;
            
            friend String operator+(const String& rStr, const Object& rObj) {
                return rStr + rObj.ToString();
            }
            
        protected:
            /**
             * The output stream for debugging
             **/
            static std::ofstream msOut;
        private:
            /**
             * Specifies if the debugging is enabled
             **/
            static bool msDebug;
            
            /**
             * Name describing the object
             **/
            String mName;
    };

}
#endif // OBJECT_H
