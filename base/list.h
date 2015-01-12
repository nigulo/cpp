#ifndef LIST_H
#define LIST_H

namespace base {

    /**
     * A pure generic list.
     * T must be primitive type or type with default 
     * constructor and assignement operator. 
     **/
    template<typename T> class List
    {
        public:
            static const long msMaxSize = 0x7fffffff;
        
    	public:
            /**
             * Initializes this list with the elements
             * taken from other list
             **/
            virtual void operator=(const List& rList);
    		
    		/**
    		 * Clears the list
    		 **/ 
    		virtual void Clear() = 0;
            
            /**
             * Appends the given list to the end of this list
             **/
            virtual void Add(const List& rList) = 0;
    		
    		/**
    		 * Returns the number of elements in the list
    		 **/
    		virtual long Size() const = 0;
    
    		/**
    		 * Adds element to the list
    		 **/
    		virtual void Add(T const & rElement) = 0;
    
    		/**
    		 * Inserts element at the given index
    		 **/
    		virtual void Insert(T const & rElement, long index) = 0;
    		
    		/**
    		 * Replaces element at the given index with the given object
    		 **/
    		virtual void Replace(T const & rElement, long index) = 0;
    
    		/**
    		 * Removes the element at the given index
    		 **/
    		virtual void RemoveAt(long index) = 0;
    
    		/**
    		 * Removes the first occurrence of the
             * given element from the list
    		 **/
    		virtual void Remove(T const & rElement);
    
            /**
             * @return the index of the given object or -1 if the object is 
             * not found. operator== must be defined for the type T.
             **/
            virtual long IndexOf(T const & rElement) const = 0;
            
            bool Contains(T const & rElement) const {
                return IndexOf(rElement) >= 0;
            }
            
    		/**
    		 * @return element at the given index
    		 **/
    		virtual T& operator[](long index) const = 0;
            
            /**
             * Swaps the elements at indices index1 and index2
             **/
            virtual void Swap(long index1, long index2) = 0;
    };
    
    /**
     * Initializes this list with the elements
     * taken from other list
     **/

    template <typename T> void List<T>::operator=(const List& rList) {
        Clear();
        Add(rList);
    }
    
	/**
	 * Removes the first occurrence of the
     * given element from the list
	 **/
	template <typename T> void List<T>::Remove(const T& rElement) {
        long index = IndexOf(rElement);
        if (index >= 0) {
            RemoveAt(index);
        }
    }

}
#endif // LIST_H
