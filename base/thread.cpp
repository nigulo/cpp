#include "thread.h"
#include <iostream>
#include <utility>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>

using namespace base;

list<Thread*> Thread::msInstances;
Mutex Thread::msMutex;

Thread::Thread()
{
    mpThread = nullptr;
}

Thread::~Thread()
{
    if (mpThread) {
    	delete mpThread;
    	msMutex.Lock();
        msInstances.remove(this);
        msMutex.Unlock();
    }
}

void Thread::Start()
{
	if (!mpThread) {
		mpThread = new thread(&Thread::Run, this);
		msMutex.Lock();
		msInstances.push_back(this);
		msMutex.Unlock();
	}
}

void Thread::Join()
{
	if (mpThread) {
		mpThread->join();
	}
}

void Thread::Sleep(int millis)
{
	this_thread::sleep_for(chrono::milliseconds(millis));
}

void Thread::Run() {}

