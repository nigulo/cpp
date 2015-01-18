#ifndef POTMANAGER_H
#define POTMANAGER_H

#include "base/object.h" // inheriting class's header file
#include "base/linkedlist.h"
#include "player.h"

namespace poker {
    class Player;
    
    /**
     * Represents the pots on the table
     * Pots consist of 1 or more sub-pots
     * (generally known as main and side pots)
     **/
    class PotManager : public Object
    {
        public:
            class Pot;
        private:

            LinkedList<Pot*> mPots;
        public:

            /**
             * Collects the chips from given players and puts
             * them into the appropriate pots
             **/
            void CollectChips(const LinkedList<Player*>& players);
            
            /**
             * Add wins to player balances
             * @return true if all pots are empty
             **/
            bool Win(LinkedList<Player*>& players);
            
            /**
             * Clears all pots
             **/
            void ClearPots();
            
            String ToString() const;
            
        public:
            class Pot : private LinkedList<Player*>
            {
                friend class PotManager;
                private:
                    int mTotalAmount;
                private:
                   
                   Pot() : mTotalAmount(0) {}
                    /**
                     * Adds given amount of chips to the pot
                     **/
                    void Add(Player& rPlayer, int amount);
                    void Add(int amount);
                    
                public:
                    
                    int TotalAmount() const {
                        return mTotalAmount;
                    }
                    
                    String ToString() const;
            };
    };
}
#endif // POTMANAGER_H
