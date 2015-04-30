#ifndef SYNCHBLOCK_H
#define SYNCHBLOCK_H

namespace base {
    /**
     * Synchronization block that can be used inside functions.
     * Use Mutex::StartSynchBlock() to create the synchronization
     * block using the given mutex object. Use Begin() to
     * start and End() to end the
     * synchronization block. If no End() is called, the assertion 
     * is generated when the SynchBlock object goes out of scope.
     * Using SynchBlock is preferred way to using Mutex::Wait() and
     * Mutex::Unlock() directly, because it guarantees that
     * Mutex::Unlock() is always called.
     * @see SynchBlock Mutex::GetSynchBlock()
     * @see Mutex::Lock()
     * @see Mutex::Unlock()
     **/
	class Mutex;

    class SynchBlock
    {
        friend class Mutex;
    	public:
    		~SynchBlock();
    		
    		/**
    		 * Starts the synchronization block.
    		 **/
    		void Begin();

    		/**
    		 * Ends the synchronization block
    		 **/
    		void End();

    	private:
    		
    		/**
    		 * Called by Mutex::GetSynchBlock()
    		 **/
    		SynchBlock(const Mutex* pMutex);
    		
    	private:
            const Mutex* mpMutex;
            bool started;
            bool ended;
    };
}
#endif // SYNCHBLOCK_H
