#ifndef MUTEX_H
#define MUTEX_H

#include <mutex>
#include "synchblock.h"
#include "object.h"

namespace base {

    class Mutex : public Object
    {
    	public:

    		/**
    		 * Creates a new mutex object.
    		 * @param initialOwner specifies if the calling thread should
             * request immediate ownership of the mutex object
    		 **/
    		Mutex();

    		// class destructor
    		~Mutex();
    		

    		void Lock() const;
    		
    		/**
    		 * Releases the ownership of the mutex
    		 **/
    		void Unlock() const;
    		
    		SynchBlock GetSynchBlock() const;

    	private:
    		std::mutex* mpMutex;
    };
}
#endif // MUTEX_H
