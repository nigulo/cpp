#ifndef HOLDEMEMULATOR_H
#define HOLDEMEMULATOR_H

#include "base/object.h" // inheriting class's header file
#include "base/linkedlist.h"
#include "card.h"
#include "holdemrules.h"

namespace poker {
    /**
     * No description
     */
    class HoldemEmulator : public Object
    {
        private:
            class CardStack {
                public:
                    static const int msNumCards = 52;
                    
                    Card* mpCards[msNumCards];
                    bool mTakenCards[msNumCards];
                    
                    CardStack() {
                        for (int i = 0; i < msNumCards; i++) {
                           	int suit = i * 4 / msNumCards;
                            int value = i % (msNumCards / 4);
                            mpCards[i] = new Card(suit, value);
                            mTakenCards[i] = false;
                        }
                    }
                    
                    ~CardStack() {
                        for (int i = 0; i < msNumCards; i++) {
                            delete mpCards[i];
                        }
                    }
                    
                    void Init() {
                        for (int i = 0; i < msNumCards; i++) {
                            mTakenCards[i] = false;
                        }
                    }
                    
            		Card* TakeRandomCard() {
            			int rnd;
            			int ind = -1;
						rnd = rand() % msNumCards;
						for (int i = rnd; i < msNumCards; i++) {
							if (!mTakenCards[i]) {
								ind = i;
								break;
							}
						}
						if (ind < 0) {
							for (int i = 0; i < rnd; i++) {
								if (!mTakenCards[i]) {
									ind = i;
									break;
								}
							}
						}
            			mTakenCards[ind] = true;
            			return mpCards[ind];
            		}
                    
    //        		Card* takeRandomCard(int value) {
    //        			int rnd;
    //        			do {
    //        				rnd = rand() % NUM_CARDS;
    //        			} while (takenCards[rnd] || cards[rnd]->value != value);
    //        			takenCards[rnd] = true;
    //        			return cards[rnd];
    //        		}
    //        
    //        		Card* takeRandomCardExcluded(Card* excludedValues[], int length) {
    //        			int rnd;
    //        			for(;;) {
    //        				rnd = rand() % NUM_CARDS;
    //        				if (takenCards[rnd]) {
    //                            continue;
    //                        }
    //                        bool ok = true;
    //        				for (int i = 0; i < length; i++) {
    //            	           //std::cout << "--" << cards[rnd]->value << "\n";
    //            	           //std::cout << "-" << excludedValues[i]->value << "\n";
    //                            if (cards[rnd]->value == excludedValues[i]->value) {
    //                                ok = false;
    //                                break;
    //                            }
    //                        }
    //                        if (ok) {
    //                            break;
    //                        }
    //        			}
    //        			takenCards[rnd] = true;
    //        			return cards[rnd];
    //        		}
            		
            		bool TakeCard(const Card* pCard) {
                        for (int i = 0; i < msNumCards; i++) {
                            if (mpCards[i]->Suit() == pCard->Suit() && mpCards[i]->Value() == pCard->Value()) {
                                if (mTakenCards[i]) {
                                    return false;
                                }
            	                mTakenCards[i] = true;
            	                return true;
                            }
                        }
            		}

            		void PutBack(const Card* pCard) {
                        for (int i = 0; i < msNumCards; i++) {
                            if (mpCards[i]->Suit() == pCard->Suit() && mpCards[i]->Value() == pCard->Value()) {
            	                mTakenCards[i] = false;
            	                break;
                            }
                        }
            		}
            		
            };
    
    	private:
            static const int msNumTests = 1000;
        
            class TwoCards {
                public:
                    Card* mpFirstCard;
                    Card* mpSecondCard;
            };
            LinkedList<TwoCards*> mStartingHands;
    	public:
    		// class constructor
    		HoldemEmulator();
    		// class destructor
    		~HoldemEmulator();
    		
            /**
             * Calculates the probability of winning with the given cards
             * before the flop
             * @return winning probability in percents
             **/
            int PreFlopWinProbability(int numPlayers, const LinkedList<Card*>& rPocketCards, int numTests = msNumTests);
            
            /**
             * Calculates the probability of winning with the given cards
             * after the flop
             * @return winning probability in percents
             **/
            int PostFlopWinProbability(int numPlayers, const LinkedList<Card*>& rPocketCards, const LinkedList<Card*>& rCommunityCards);
    };
}
#endif // HOLDEMEMULATOR_H
