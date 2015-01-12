#include "mutex.h" // class's header file

using namespace base;
using namespace std;

/**
 * Creates a new mutex object.
 * @param initialOwner specifies if the calling thread should
 * request immediate ownership of the mutex object
 **/
Mutex::Mutex()
{
    mpMutex = new mutex();
}

// class destructor
Mutex::~Mutex()
{
	if (mpMutex) {
        delete mpMutex;
    }
}

void Mutex::Lock() const
{
    if (mpMutex) {
    	mpMutex->lock();
    }
    return;
}

void Mutex::Unlock() const
{
    if (mpMutex) {
        mpMutex->unlock();
    }
}

SynchBlock Mutex::GetSynchBlock() const
{
    return SynchBlock(this);
}
