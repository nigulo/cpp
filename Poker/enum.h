#ifndef ENUM_H
#define ENUM_H

namespace poker {
    /**
     * No description
     */
    template<typename T> class Enum
    {
    	private:
            T mValue;
    	protected:
    		Enum() {
            }
    		// class constructor
    		Enum(const T value) : mValue(value) {
            }
    	public:
            bool operator==(const Enum<T>& rOther) {
                return mValue == rOther.mValue;
            }
            
            T Value() const {
                return mValue;
            }
    };
}
#endif // ENUM_H
