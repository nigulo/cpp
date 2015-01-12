#include <cassert>
#include "list.h"

#ifndef LINKEDLIST_H
#define LINKEDLIST_H

using namespace std;

namespace base {

    /**
     * Generic linkedlist.
     * Type T must be pointer or 
     * class with default constructor, assignment and == operators.
     * This implementation is unsynchronized.
     */
    template <typename T> class LinkedList : public List<T>
    {

        private:
            class ListElement
            {
                public:
                    T mElement;
                    ListElement* mpNextElement;
                    ListElement* mpPrevElement;
                public:
                    ListElement(T const & rElement) :
                        mElement(rElement),
                        mpNextElement(0),
                        mpPrevElement(0)
                        {
                    }
                    
                    ~ListElement() {
                    }
            };

        public:
            class Iterator //: public List<T>::Iterator
            {
                friend class LinkedList;

                private:
                    ListElement* mpElement;

                private:
                    Iterator() {
                        mpElement = 0;
                    }
                    
                    Iterator(ListElement* pElement) {
                        mpElement = pElement;
                    }
    
                public:
                    bool operator==(const Iterator& iter) const {
                        assert(iter.mpElement);
                        return mpElement == iter.mpElement;
                    }
    
                    bool operator!=(const Iterator& iter) const {
                        return mpElement != iter.mpElement;
                    }
    
                    void operator++() {
                        assert(mpElement);
                        if (!mpElement->mpNextElement) {
                            mpElement = 0;
                        }
                        else {
                            //assert(iter.mpElement->mpNextElement);
                            mpElement = mpElement->mpNextElement;
                        }
                    }
    
                    void operator--() {
                        assert(mpElement);
                        if (!mpElement->mpPrevElement) {
                            mpElement = 0;
                        }
                        else {
                            //assert(iter.mpElement->mpPrevElement);
                            mpElement = mpElement->mpPrevElement;
                        }
                    }
                    
                    void operator++(int) {
                        assert(mpElement);
                        if (!mpElement->mpNextElement) {
                            mpElement = 0;
                        }
                        else {
                            //assert(mpElement->mpNextElement);
                            mpElement = mpElement->mpNextElement;
                        }
                    }
    
                    void operator--(int) {
                        assert(mpElement);
                        if (!mpElement->mpPrevElement) {
                            mpElement = 0;
                        }
                        else {
                            //assert(mpElement->mpPrevElement);
                            mpElement = mpElement->mpPrevElement;
                        }
                    }
    
                    /**
                     * @return reference to the element this iterator
                     * is currently pointing to
                     **/
                    T& operator*() const {
                        assert(mpElement);
                        return mpElement->mElement;
                    }
    
                    /**
                     * @return true if this iterator points to nothing
                     **/
                    bool Done() const {
                        return !mpElement;
                    }
            };

    	public:
    		// class constructor
    		LinkedList() : mSize(0) {
                mpFirstElement = 0;
                mpLastElement = 0;
            }
    
    		//LinkedList(const LinkedList& list) {
            //    Add(list);
            //}
    
    		// class destructor
    		virtual ~LinkedList() {
                if (mpFirstElement) {
                    delete mpFirstElement;
                }
            }
    		

    		/**
    		 * @return the begin iterator of this linkedlist
             **/
            Iterator Begin() const {
                return mBegin;
            }
    
    		/**
    		 * @return the end iterator of this linkedlist
             **/
    		Iterator End() const {
                return mEnd;
            }
    		
    		/**
    		 * Clears the linkedlist
    		 **/ 
    		void Clear();
            
            /**
             * Appends the given list to the end of this linkedlist
             **/
            void Add(const List<T>& rList);
    		
    		/**
    		 * Returns the number of elements in the linkedlist
    		 **/
    		long Size() const {
                return mSize;
            }
    
    		/**
    		 * Adds element to the linkedlist
    		 **/
    		void Add(T const & rElement);
    
    		/**
    		 * Inserts element at the given index
    		 **/
    		void Insert(T const & rElement, long index);
    		
    		/**
    		 * Replaces element at the given index with the given object
    		 **/
    		void Replace(T const & rElement, long index);
    
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
             * Number of elements in the linkedlist
             **/
            long mSize;
            /**
             * First element in the linkedlist or NULL if the linkedlist is empty
             **/
            ListElement* mpFirstElement;
            /**
             * Last element in the linkedlist or NULL if the linkedlist is empty.
             * mpFirstElement = mpLastElement is the linkedlist contains the
             * single element
             **/
            ListElement* mpLastElement;
            /**
             * Begin iterator of the linkedlist or NULL if the linkedlist is empty
             **/
            Iterator mBegin;
            /**
             * End iterator of the linkedlist or NULL if the linkedlist is empty.
             * mpBegin = mpEnd is the linkedlist contains the
             * single element
             **/
            Iterator mEnd;
    };
    
	/**
	 * Clears the linkedlist
	 **/ 
	template <typename T> void LinkedList<T>::Clear() {
        if (mpFirstElement) {
            delete mpFirstElement;
        }
        mBegin.mpElement = 0;
        mEnd.mpElement = 0;
        mpFirstElement = 0;
        mpLastElement = 0;
        mSize = 0;
    }
    
    /**
     * Appends the given list to the end of this linkedlist
     **/
    template <typename T> void LinkedList<T>::Add(const List<T>& rList) {
        for (long i = 0; i < rList.Size(); i++) {
            Add(rList[i]);
        }
    }
    
    /**
     * Adds element to the linkedlist
     **/
    template <typename T> void LinkedList<T>::Add(T const & rElement) {
        assert(mSize < List<T>::msMaxSize);
        if (!mpFirstElement) {
            mpFirstElement = new ListElement(rElement);
            mpLastElement = mpFirstElement;
        }
        else {
            ListElement* p_element = mpFirstElement;
            while (p_element->mpNextElement) {
                p_element = p_element->mpNextElement;
            }
            mpLastElement = new ListElement(rElement);
            mpLastElement->mpPrevElement = p_element;
            p_element->mpNextElement = mpLastElement;
        }
        mBegin.mpElement = mpFirstElement;
        mEnd.mpElement = mpLastElement;
        mSize++;
        //-----------------------------
        // Debug
        //std::cout << "size: " << mSize << '\n';
        //-----------------------------
    }
    
    /**
     * Inserts element at the given index. If index is equal to the size of
     * the linkedlist, the element is added to the end of the linkedlist.
     **/
    template <typename T> void LinkedList<T>::Insert(T const& rElement, long index)
    {
        assert(index >= 0 && index <= mSize);
        assert(mSize < List<T>::msMaxSize);
        if (!mpFirstElement) {
            mpFirstElement = new ListElement(rElement);
            mpLastElement = mpFirstElement;
        }
        else {
            ListElement* p_element = mpFirstElement;
            for (long i = 0; i < index - 1; i++) {
                p_element = p_element->mpNextElement;
            }
            ListElement* p_new_element = new ListElement(rElement);
            p_new_element->mpNextElement = p_element->mpNextElement;
            p_new_element->mpPrevElement = p_element;
            p_element->mpNextElement = p_new_element;
            if (!p_new_element->mpNextElement) {
                mpLastElement = p_new_element;
            }
        }
        mBegin.mpElement = mpFirstElement;
        mEnd.mpElement = mpLastElement;
        mSize++;
        //-----------------------------
        // Debug
        //std::cout << "size: " << mSize << '\n';
        //-----------------------------
    }
    
    /**
     * Replaces element at the given index with the given object
     **/
    template <typename T> void LinkedList<T>::Replace(T const & rElement, long index)
    {
        assert(index >= 0 && index < mSize);
        ListElement* p_element = mpFirstElement;
        for (long i = 0; i < index; i++) {
            p_element = p_element->mpNextElement;
        }
        // Reassign old object
        p_element->mElement = rElement;
    }
    
    /**
     * Removes the element at the given index and returns it
     **/
    template <typename T> void LinkedList<T>::RemoveAt(long index)
    {
        assert(index >= 0 && index < mSize);
        ListElement* p_element = mpFirstElement;
        for (long i = 0; i < index; i++) {
            p_element = p_element->mpNextElement;
        }
        // Get the pointer to the next element
        ListElement* p_next_element = p_element->mpNextElement;
        ListElement* p_prev_element = p_element->mpPrevElement;
        if (p_prev_element) {
            //std::cout << "prev elem" << '\n';
            // If the element to remove is not the first one
            // reassign the mpNextElement of the previous
            // element
            p_prev_element->mpNextElement = p_next_element;
        }
        if (p_next_element) {
            //std::cout << "next elem" << '\n';
            p_next_element->mpPrevElement = p_prev_element;
        }
        if (p_element == mpFirstElement) {
            //std::cout << "removed first elem" << '\n';
            mpFirstElement = p_next_element;
        }
        if (p_element == mpLastElement) {
            //std::cout << "removed last elem" << '\n';
            mpLastElement = p_prev_element;
        }
        delete p_element;
        mBegin.mpElement = mpFirstElement;
        mEnd.mpElement = mpLastElement;
        mSize--;
        //-----------------------------
        // Debug
        //std::cout << "size: " << mSize << '\n';
        //-----------------------------
    }
    
    /**
     * @return the index of the given object or -1 if the object is 
     * not found. operator== must be defined for the type T.
     **/
    template <typename T> long LinkedList<T>::IndexOf(T const & rElement) const
    {
        if (!mpFirstElement) {
            return -1;
        }
        ListElement* p_element = mpFirstElement;
        if (p_element->mElement == rElement) {
            return 0;
        }
        long index = 0;
        while (p_element->mpNextElement) {
            p_element = p_element->mpNextElement;
            index++;
            if (p_element->mElement == rElement) {
                return index;
            }
        }
        return -1;
    }
    
    /**
     * @eturn element at the given index
     **/
    template <typename T> T& LinkedList<T>::operator[](long index) const {
        assert(index>= 0 && index < mSize);
        ListElement* p_element = mpFirstElement;
        for (long i = 0; i < index; i++) {
            p_element = p_element->mpNextElement;
        }
        return p_element->mElement;
    }
    
    /**
     * Swaps the elements at indices index1 and index2
     **/
    template <typename T> void LinkedList<T>::Swap(long index1, long index2) {
        assert(index1 >= 0 && index2 >= 0 && index1 < mSize && index2 < mSize);
        if (index1 != index2) {
            ListElement* p_element_1 = mpFirstElement;
            for (long i = 0; i < index1; i++) {
                p_element_1 = p_element_1->mpNextElement;
            }
            ListElement* p_element_2 = mpFirstElement;
            for (long i = 0; i < index2; i++) {
                p_element_2 = p_element_2->mpNextElement;
            }
            T& r_element = p_element_1->mElement;
            p_element_1->mElement = p_element_2->mElement;
            p_element_2->mElement = r_element;
        }
    }

}
#endif // LINKEDLIST_H
