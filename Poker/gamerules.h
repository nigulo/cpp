#ifndef GAMERULES_H
#define GAMERULES_H

#include "base/object.h"
#include "base/linkedlist.h"
#include "cardstack.h"
#include "player.h"

namespace poker {
    
    class Table;
    
    /**
     * No description
     */
    class GameRules : public Object
    {
        public:
            class Winner {
                Player& mrPlayer;

            };
            
        protected:
            Table& mrTable;
    	public:
            GameRules(Table& rTable) : mrTable(rTable) {}

            virtual void DealPocketCards() = 0;
            virtual void DealFlop() = 0;
            virtual void DealTurn() = 0;
            virtual void DealRiver() = 0;
    		
    		virtual void DetermineWinners() = 0;
    		
        public:    		
            static long GetHandStrength(Card* cards[], int length);
        //private:
            
            static long HasPair(Card* cards[], int length);
            static long HasThreeOfAKind(Card* cards[], int length);
            static long HasFourOfAKind(Card* cards[], int length);
            static long HasTwoPair(Card* cards[], int length);
            static long HasFullHouse(Card* cards[], int length);
            static long HasStraight(Card* cards[], int length);
            static long HasFlush(Card* cards[], int length);
            static long HasStraightFlush(Card* cards[], int length);
            static long GetHandPoints(Card* cards[], bool used[], int length);

            static int CountEquals(int value, Card* cards[], int length);
            static bool HasPair(int value, Card* cards[], int length);
            static bool HasThreeOfAKind(int value, Card* cards[], int length);
            static bool HasFourOfAKind(int value, Card* cards[], int length);
            static bool HasTwoPair(int value1, int value2, Card* cards[], int length);
            static bool HasFullHouse(int value1, int value2, Card* cards[], int length);

            /**
             * @return the maximum number of suited cards
             **/
            static int GetMaxNumSuited(Card* cards[], int length, Card* cardsOut[]);
            
            /**
             * Sorts the cards
             **/                		
            static void SortCards(Card* cards[], int length, int indices[]);
    };
}
#endif // GAMERULES_H
