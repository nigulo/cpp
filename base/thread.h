#ifndef THREAD_H
#define THREAD_H

#include <thread>
#include "object.h"
#include "arraylist.h"
#include "mutex.h"

namespace base {
    /**
     * No description
     */
    class Thread : public Object
    {
    	public:
    		// class constructor
    		/**
             * @param stackSize the initial stack size for the thread
             **/
    		Thread();
    		
    		//Thread(void (*pThreadFunc)(void*));
    		
    		// class destructor
    		virtual ~Thread();
 		
            /**
             * Starts the thread.
             **/
    		void Start();

    		void Join();

    		static void Sleep(int millis);
            /**
             * This method is called by the system in a separate thread.
             **/
    		virtual void Run();

    	private:

            /**
             * Thread handle
             **/
            std::thread* mpThread;
            
            /**
             * All thread instances of the application
             **/
            static LinkedList<Thread*> msInstances;
             
             /**
              * All methods accessing msInstances are
              * synchronized using this member
              **/
            static Mutex msMutex;
    };
}

#endif // THREAD_H
