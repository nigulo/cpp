#include <iostream>
#include <utility>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>
#include "thread.h" // class's header file

using namespace base;

LinkedList<Thread*> Thread::msInstances;
Mutex Thread::msMutex;

// class constructor
Thread::Thread()
{
    mpThread = nullptr;
}

// class destructor
Thread::~Thread()
{
    if (mpThread) {
    	delete mpThread;
    	msMutex.Lock();
        msInstances.Remove(this);
        msMutex.Unlock();
    }
}

void Thread::Start()
{
	if (!mpThread) {
		mpThread = new thread(&Thread::Run, this);
		msMutex.Lock();
		msInstances.Add(this);
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

