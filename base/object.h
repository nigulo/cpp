#ifndef OBJECT_H
#define OBJECT_H

#include <fstream>
#include <string>

using namespace std;

namespace base {
    /*
     * Base class of all objects
     */
    class Object
    {
    	public:
			/**
			 * Constructs an object with the given name
			 **/
			Object(const string& rName = "");

			/**
    		 * Copy constructor
    		 **/
    		Object(const Object& rObj);

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
    		const string& Name() const {
            	return mName;
            }

            /**
             * Prints the given text prefixed with the
             * object's name to output stream if the debugging
             * is enabled.
             **/            
    		virtual void Debug(const string& rText) const;

            /**
             * Prints the given text to output stream if 
             * the debugging is enabled.
             **/            
            static void Dbg(const string& rText);
            
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
            virtual string ToString() const;
            
            friend string operator+(const string& rStr, const Object& rObj) {
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
            string mName;
    };

}
#endif // OBJECT_H
