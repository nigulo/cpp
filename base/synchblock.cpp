#include "synchblock.h"
#include "mutex.h"
#include <cassert>

using namespace base;

SynchBlock::SynchBlock(const Mutex* pMutex) : 
    started(false),
    ended(false)
{
    mpMutex = pMutex;
}

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
