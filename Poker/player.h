#ifndef PLAYER_H
#define PLAYER_H

#include "base/object.h" // inheriting class's header file
#include "base/string.h"
#include "base/linkedlist.h"
#include "action.h"
#include "card.h"

namespace poker {
    /**
     * No description
     **/
    class Table;
    class Seat;
    
    class Player : public Object
    {
        
        private:
            String mName;
            int mChips;
            int mCurrentBet;
            int mTotalBet;
            int mWin;
            LinkedList<Card*> mCards;
            //LinkedList<Card*>* mpHand;
            unsigned long mHandStrength;
            Seat& mrSeat;
            bool mSitIn;
            
            /**
             * 0 - normal, 49 - very tight, -50 - very loose
             **/
            int mTightness;
            //------------------------------------
            
            /**
             * Actions available to the player
             **/
            LinkedList<Action*> mAvailableActions;
            
            /**
             * Actions made by player in the
             * current round
             **/
            LinkedList<Action*> mRoundActions;

            /**
             * Actions made by player in the
             * current round
             **/
            LinkedList<Action*> mActions;

            /**
             * Has the player already made an action
             **/            
            bool mActive;
            
        public:
            Player(const String& rName,  Seat& rSeat, int initalChips = 0);
            
            virtual ~Player();

            String ToString() const;
            
            const String& Name() const {
                return mName;
            }
            
            void BuyChips(int amount);
            
            int Chips() const {
                return mChips;
            }

            int CurrentBet() const {
                return mCurrentBet;
            }
            
            int TotalBet() const {
                return mTotalBet;
            }

            const Seat& GetSeat() const {
                return mrSeat;
            }
            
            bool IsSitIn() const {
                return mSitIn;
            }

            bool IsAllin() const;

            bool IsFolded() const;
           
            void Win(int amount);
            
            /**
             * Player takes an action (from
             * the available actions list)
             **/
            void TakeAction(Action& rAction);
            
            /**
             * Adds an action to the available actions list
             **/
            
            void AddAvailableAction(const Action& rAction);

            /**
             * @return the last action made by player
             **/
    		Action* GetLastAction() const;
    		
            /**
             * @return the last action made by player in
             * the current round
             **/
    		Action* GetLastRoundAction() const;

    		void GameStarted();
    		void RoundFinished();
    		
    		void AddCard(Card* pCard);
    		/**
    		 * Notifies the player about getting focus
    		 **/
    		void AskAction();
    		
    		bool IsActive() const {
                return mActive;
            }
            
            //void SetHand(LinkedList<Card*>* pHand) {
            //    mpHand = pHand;
            //}

            void SetHandStrength(unsigned long strength) {
                mHandStrength = strength;
            }

            unsigned long HandStrength() const {
                return mHandStrength;
            }
            
            const LinkedList<Card*>& GetCards() const {
                return mCards;
            }
            
            const LinkedList<Action*>& GetAvailableActions() const {
                return mAvailableActions;
            }
            
            int GetTightness() const {
                return mTightness;
            }
    };
}
#endif // PLAYER_H
