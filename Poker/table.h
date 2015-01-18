#ifndef TABLE_H
#define TABLE_H

#include "base/thread.h"
#include "base/linkedlist.h"
#include "card.h"
#include "cardstack.h"
#include "gamerules.h"
#include "enum.h"
#include "potmanager.h"
#include "action.h"
#include "player.h"
#include "seat.h"
#include "cardstack.h"

namespace poker {
    
    class Player;
    
    /**
     * No description
     */
    class Table : public Thread
    {
        public:
            
            class GameStage;
            class PreFlopStage;

            static const int msSmallBlind = 5;
            static const int msBigBlind = 10;

        private:

            static const int msNumSeats = 10;
            static const int msMaxRaises = 3;
            
            GameRules* mpGameRules;

            /**
             * Seats around the table
             **/
            Seat* mpSeats[msNumSeats];
            
            /**
             * Players in the running hand
             **/
            LinkedList<Player*> mPlayers;
            
            /**
             * Community cards
             **/
            LinkedList<Card*> mCommunityCards;
            
            /**
             * The card stack
             **/
            CardStack mCards;
            
            /**
             * Pots
             **/
            PotManager mPotManager;;
                         
            Player* mpDealer;
            Player* mpCurrentPlayer;
             
            /**
             * The action histroy
             **/
            LinkedList<Action*> mActions;

            /**
             * The action histroy of the current round
             **/
            LinkedList<Action*> mRoundActions;
             
            /**
             * Game stage
             **/
            GameStage* mpGameStage;
              
            /**
             * Current bet on the table
             **/
            int mCurrentBet;
              
            /**
             * Number of raises done in the current round
             **/
            int mNumRaises;

            Mutex* mpMutex;

               
    	public:
    		// class constructor
    		Table(Mutex* pMutex);
    		// class destructor
    		~Table();
    		
    		String ToString() const;
    		
    		void Run();
    		
    		void TakeAction(Player& rPlayer, Action& rAction);
    		Player* GetActivePlayer();

    		int CurrentBet() const {
                return mCurrentBet;
            }

            /**
             * @return the last action made on table
             **/
    		Action* GetLastAction() const;

            /**
             * @return the last action int the current round 
             * made on table
             **/
    		Action* GetLastRoundAction() const;
    		
    		/**
    		 * Adds one community card
    		 **/
    		void AddCommunityCard(Card* pCard);
    		
    		LinkedList<Player*>& GetPlayers() {
                return mPlayers;
            }

    		CardStack& GetCardStack() {
                return mCards;
            }

    		const LinkedList<Card*>& GetCommunityCards() const {
                return mCommunityCards;
            }

            /**
             * For HoldemRules only
             **/
    		PotManager& GetPotManager() {
                return mPotManager;
            }
            
            /**
             * @return total bet on table. The return value 
             * includes all pots and bets in the current round
             **/
            int TotalBet() const;
            
            /**
             * @return first free seat or 0 if no free seats are
             * available
             **/
            
            Seat* FirstFreeSeat() const;
            
            const GameStage* GetGameStage() const {
                return mpGameStage;
            }
            
            int CountPlayers() const;
            
            /**
             * Called by HoldemRules after cards are dealt
             **/
            void RenderCards(const String& rText) const;

    		/**
    		 * @return the allowed bet value in the current round.
    		 * Returns BB in preflop and flop, 2 * BB in turn
    		 * and river stages.
    		 **/
    		int BetValue() const;
    		
    		Player* GetDealer() const {
                return mpDealer;
            }

    	private:
            
            void Play();
            
            void StartGame();
            void PreFlop();
            void Flop();
            void Turn();
            void River();
            void Finish();
    		
    		void MoveDealer();
    		
    		/**
    		 * Blind postings
    		 **/
    		void Blinds();
    		
    		/**
    		 * Betting round actions
    		 **/
    		void BettingRound();
    		
    		/**
    		 * Checks if the current betting round is finished
    		 **/
    		bool RoundFinished() const;

    		/**
    		 * Checks if the game is finished
    		 **/
    		bool GameFinished() const;
    		
    		void MakeAction(Action& rAction);

    		void NextRound();

    		/**
    		 * @return either next empty or taken seat counted 
             * from index. If the end of the seat array is 
             * reached the  search is continued from the beginning. 
             * If no taken seat is found the return value is 0.
    		 **/
    		Seat* GetNextSeat(bool taken, int index = 0) const;
    		
    		/**
    		 * @return either next player counted from index.
    		 * @param sitIn specifies if the returned player must be
    		 * sitting in
    		 * @param index specifies the start index of the search
    		 * @param allin specifies if the returned player may be 
             * allin
    		 * @param folded specifies if the returned player may be 
             * folded
             * If the end of the seat array is 
             * reached the  search is continued from the beginning. 
             * If no such player is found the return value is 0.
    		 **/
    		Player* GetNextPlayer(bool sitIn, int index = 0, bool allin = false, bool folded = false) const;
    		
    		/**
    		 * @return next player counted from current player.
             * The player returned is sitting in, is neither allin
             * nor folded. 
             * If the end of the seat array is 
             * reached the  search is continued from the beginning. 
             * If no such player is found the return value is 0.
    		 **/
    		Player* GetNextPlayer() const;

    		/**
    		 * @return next player counted from given player.
             * The player returned is sitting in, is neither allin
             * nor folded. 
             * If the end of the seat array is 
             * reached the  search is continued from the beginning. 
             * If no such player is found the return value is 0.
    		 **/
    		Player* GetNextPlayer(const Player& rPlayer) const;
    		
    		
        // Inner Classes -------------------
    	public:
            class GameStage : public Object
            {
                friend class Table;
                friend class PreFlopStage;
                public:
                    class Type : public Enum<int>
                    {
                        friend class GameStage;
                        friend class PreFlopStage;
                        public:
                            static const Type msStart;
                            static const Type msFlop;
                            static const Type msTurn;
                            static const Type msRiver;
                            static const Type msFinish;
                        private:
                            static const Type msPreFlop;
                    };
                    
                protected:
                    GameStage() :
                        mrType(Type::msStart),
                        mCardsDealt(false)
                    {
                    }

                    GameStage(const Type& rType) :
                        mrType(rType),
                        mCardsDealt(false)
                        {
                        }

                    const Type& mrType;
                    bool mCardsDealt;
                    
                public:
                    const Type& GetType() const {
                        return mrType;
                    }
                    
            };
            
            class PreFlopStage : public GameStage
            {
                friend class Table;
                private:
                    bool mDealerMoved;
                    bool mSmallBlindPosted;
                    bool mBigBlindPosted;
                private:
                    PreFlopStage() :
                        mDealerMoved(false),
                        mSmallBlindPosted(false),
                        mBigBlindPosted(false),
                        GameStage(Type::msPreFlop)
                    {
                    }
            };
    };
}
#endif // TABLE_H
