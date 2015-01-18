#ifndef CARDSTACK_H
#define CARDSTACK_H

#include "card.h"

namespace poker {
    class CardStack : public Object
    {
        public:
            static const int msNumCards = 52;
    
        private:        
            Card* mpCards[msNumCards];
            int mIndex;
            
        public:
            CardStack();
            
            ~CardStack();
            
            /**
             * Marks all cards as available in the stack
             **/
            void Shuffle();
            
    		/**
    		 * Marks the random card as taken in the stack 
             * and returns the pointer to the card
    		 **/
    		Card* DealCard();
            
    		/**
    		 * Marks the random card with given value as 
             * taken in the stack and returns the pointer
             * to the card
    		 **/
    		//Card* TakeRandomCard(int value);
    
    		//Card* TakeRandomCardExcluded(Card* pExcludedValues[], int length);
    		
    		/**
    		 * Marks the given card as taken in the stack
    		 **/
    		//void TakeCard(const Card* pCard);
    };
}
#endif // CARDSTACK_H
