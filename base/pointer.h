#ifndef POINTER_H
#define POINTER_H

namespace base {
    template <typename T> class Pointer;
    /**
     * Smart pointer. If this object goes out of scope,
     * underlying object is deleted
     */
    template <typename T> class Pointer<T*>
    {
    	public:
    		// class constructor
    		Pointer(T* pPointer) {
                mpPointer = pPointer;
            }
    		// class destructor
    		~Pointer() {
                delete mpPointer;
            }
            
    		T* operator*() const {
                return mpPointer;
            }
        private:
            T* mpPointer;
    };
}
#endif // POINTER_H
