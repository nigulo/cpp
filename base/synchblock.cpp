#include "synchblock.h" // class's header file
#include "mutex.h"

using namespace base;

// class constructor
SynchBlock::SynchBlock(const Mutex* pMutex) : 
    started(false),
    ended(false)
{
    mpMutex = pMutex;
}

// class destructor
SynchBlock::~SynchBlock()
{
    assert((!started && !ended) || (started && ended));
}

void SynchBlock::Begin()
{
    started = true;
    mpMutex->Lock();
}

void SynchBlock::End()
{
    mpMutex->Unlock();
    ended = true;
}
