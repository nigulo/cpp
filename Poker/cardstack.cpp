#include <stdlib.h>
#include <time.h>
#include "cardstack.h" // class's header file

using namespace poker;

CardStack::CardStack() : mIndex(0) {
    for (int i = 0; i < msNumCards; i++) {
       	int suit = i * 4 / msNumCards;
        int value = i % (msNumCards / 4);
        mpCards[i] = new Card(suit, value);
        //Debug(String("Creating cardstack: ") + mpCards[i]->ToString());
    }
}

CardStack::~CardStack() {
    for (int i = 0; i < msNumCards; i++) {
        delete mpCards[i];
    }
}

/**
 * Marks all cards as available in the stack
 **/
void CardStack::Shuffle() {
	time_t systime;
	time(&systime);
	srand(systime);
    for (int i = 0; i < msNumCards; i++) {
		int rnd = rand() % msNumCards;
		Card* p_card = mpCards[rnd];
		mpCards[rnd] = mpCards[i];
		mpCards[i] = p_card;
    }
    //for (int i = 0; i < msNumCards; i++) {
	//	Debug(String("Card: ") + mpCards[i]->ToString());
    //}
    mIndex = 0;
}

/**
 * Marks the random card as taken in the stack 
 * and returns the pointer to the card
 **/
Card* CardStack::DealCard() {
    assert(mIndex < msNumCards);
    return(mpCards[mIndex++]);
}

/**
 * Marks the random card with given value as 
 * taken in the stack and returns the pointer
 * to the card
 **/
//Card* CardStack::TakeRandomCard(int value) {
//	int rnd;
//	do {
//		rnd = rand() % msNumCards;
//	} while (mTakenCards[rnd] || mpCards[rnd]->Value() != value);
//	mTakenCards[rnd] = true;
//	return mpCards[rnd];
//}
//
//Card* CardStack::TakeRandomCardExcluded(Card* pExcludedValues[], int length) {
//	int rnd;
//	for(;;) {
//		rnd = rand() % msNumCards;
//		if (mTakenCards[rnd]) {
//            continue;
//        }
//        bool ok = true;
//		for (int i = 0; i < length; i++) {
//           //std::cout << "--" << cards[rnd]->value << "\n";
//           //std::cout << "-" << excludedValues[i]->value << "\n";
//            if (mpCards[rnd]->Value() == pExcludedValues[i]->Value()) {
//                ok = false;
//                break;
//            }
//        }
//        if (ok) {
//            break;
//        }
//	}
//	mTakenCards[rnd] = true;
//	return mpCards[rnd];
//}

/**
 * Marks the given card as taken in the stack
 **/
//void CardStack::TakeCard(const Card* pCard) {
//    for (int i = 0; i < msNumCards; i++) {
//        if (mpCards[i]->Suit() == pCard->Suit() && mpCards[i]->Value() == pCard->Value()) {
//            mTakenCards[i] = true;
//            break;
//        }
//    }
//}
