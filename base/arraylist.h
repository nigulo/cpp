#ifndef ARRAYLIST_H
#define ARRAYLIST_H

#include "list.h" // inheriting class's header file
#include <cassert>
using namespace std;

namespace base {

    /**
     * A generic arraylist.
     * T must be primitive type or type with default 
     * constructor and assignement operator. 
     **/
    template <typename T> class ArrayList : public List<T>
    {

    	private:
            T* mpElements;
            long mReAllocateBy;
            long mSize;
            long mCapacity;
        
    	public:
    		// class constructor
    		ArrayList(long initialCapacity = 0, long reAllocateBy = 10) : 
                mpElements(0),
                mReAllocateBy(reAllocateBy),
                mSize(0),
                mCapacity(initialCapacity)
            {
                assert(reAllocateBy > 0 && initialCapacity >= 0);
                if (initialCapacity > 0) {
                    mpElements = new T[initialCapacity];
                }
            }
            
    		// class destructor
    		virtual ~ArrayList() {
                Clear();
            }

    		/**
    		 * Clears the list
    		 **/ 
    		void Clear() {
                if (mpElements) {
                    delete []mpElements;
                }
                mSize = 0;
                mCapacity = 0;
            }
            
            /**
             * Appends the given list to the end of this list
             **/
            void Add(const List<T>& rList);
    		
    		/**
    		 * Returns the number of elements in the list
    		 **/
    		long Size() const {
                return mSize;
            }
    
    		/**
    		 * Adds element to the list
    		 **/
    		void Add(T const & rElement);
    
    		/**
    		 * Inserts element at the given index
    		 **/
    		void Insert(T const & rElement, long index);
    		
    		/**
    		 * Replaces element at the given index with the given object
    		 **/
    		void Replace(T const & rElement, long index) {
                mpElements[index] = rElement;
            }
    
    		/**
    		 * Removes the element at the given index
    		 **/
    		void RemoveAt(long index);
    
            /**
             * @return the index of the given object or -1 if the object is 
             * not found. operator== must be defined for the type T.
             **/
            long IndexOf(T const & rElement) const;
            
    		/**
    		 * @return element at the given index
    		 **/
    		T& operator[](long index) const;
            
            /**
             * Swaps the elements at indices index1 and index2
             **/
            void Swap(long index1, long index2);
            
        private:
            
            /**
             * Reallocates the array if needed
             **/
            void ReAllocate();
            
    };

    /**
     * Appends the given list to the end of this list
     **/
    template <typename T> void ArrayList<T>::Add(const List<T>& rList) {
        for (long i = 0; i < rList.Size(); i++) {
            Add(rList[i]);
        }
    }

	/**
	 * Adds element to the list
	 **/
	template <typename T> void ArrayList<T>::Add(T const & rElement) {
        assert(mSize < List<T>::msMaxSize);
        ReAllocate();
        mpElements[mSize++] = rElement;
        //-----------------------------
        // Debug
        //std::cout << "size: " << mSize << '\n';
        //std::cout << "capacity: " << mCapacity << '\n';
        //std::cout << "increment by: " << mReAllocateBy << '\n';
        //-----------------------------
    }

	/**
	 * Inserts element at the given index
	 **/
	template <typename T> void ArrayList<T>::Insert(T const & rElement, long index) {
        assert(mSize < List<T>::msMaxSize);
        if (index == mSize) {
            Add(rElement);
            return;
        }
        assert(index >= 0 && index < mSize);
        ReAllocate();
        for (long i = mSize - 1; i >= index; i--) {
            mpElements[i + 1] = mpElements[i];
        }
        mpElements[index] = rElement;
        mSize++;
        //-----------------------------
        // Debug
        //std::cout << "size: " << mSize << '\n';
        //std::cout << "capacity: " << mCapacity << '\n';
        //std::cout << "increment by: " << mReAllocateBy << '\n';
        //-----------------------------
    }

	/**
	 * Removes the element at the given index
	 **/
	template <typename T> void ArrayList<T>::RemoveAt(long index) {
        assert(index >= 0 && index < mSize);
        for (long i = index ; i < mSize - 1; i++) {
            mpElements[i] = mpElements[i + 1];
        }
        mSize--;
        ReAllocate();
        //-----------------------------
        // Debug
        //std::cout << "size: " << mSize << '\n';
        //std::cout << "capacity: " << mCapacity << '\n';
        //std::cout << "increment by: " << mReAllocateBy << '\n';
        //-----------------------------
    }
    
    /**
     * @return the index of the given object or -1 if the object is 
     * not found. operator== must be defined for the type T.
     **/
    template <typename T> long ArrayList<T>::IndexOf(T const & rElement) const {
        for (long i = 0; i < mSize; i++) {
            if (mpElements[i] == rElement) {
                return i;
            }
        }
        return -1;
    }

	/**
	 * @return element at the given index
	 **/
	template <typename T> T& ArrayList<T>::operator[](long index) const {
        assert(index < mSize);
        return mpElements[index];
    }

    /**
     * Swaps the elements at indices index1 and index2
     **/
    template <typename T> void ArrayList<T>::Swap(long index1, long index2) {
        assert(index1 >= 0 && index2 >= 0 && index1 < mSize && index2 < mSize);
        if (index1 != index2) {
            T element = mpElements[index1];
            mpElements[index1] = mpElements[index2];
            mpElements[index2] = element;
        }
    }

    /**
     * Reallocates the array if needed
     **/
    template <typename T> void ArrayList<T>::ReAllocate()
    {
        if (mSize >= mCapacity) {
            T* p_elements = new T[mCapacity + mReAllocateBy];
            for (long i = 0; i < mSize; i++) {
                p_elements[i] = mpElements[i];
            }
            T* p_old_elements = mpElements;
            mpElements = p_elements;
            mCapacity += mReAllocateBy;
            delete []p_old_elements;
        }
        else if (mSize <= mCapacity - mReAllocateBy) {
            T* p_elements = new T[mCapacity - mReAllocateBy];
            for (long i = 0; i < mSize; i++) {
                p_elements[i] = mpElements[i];
            }
            T* p_old_elements = mpElements;
            mpElements = p_elements;
            mCapacity -= mReAllocateBy;
            delete []p_old_elements;
        }
    }
}
#endif // ARRAYLIST_H
