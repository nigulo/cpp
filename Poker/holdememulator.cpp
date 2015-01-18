#include <stdlib.h>

#include "holdememulator.h" // class's header file

using namespace poker;

// class constructor
HoldemEmulator::HoldemEmulator()
{
    cout << "Generating starting hands...\n";
	for (int i = 0; i < 1000; i++) {
    	CardStack card_stack;
    	card_stack.Init();
        Card* p_card1 = card_stack.TakeRandomCard();
        Card* p_card2 = card_stack.TakeRandomCard();
        LinkedList<Card*> cards;
        cards.Add(p_card1);
        cards.Add(p_card2);
        if (PreFlopWinProbability(10, cards, 100) > 10) {
            TwoCards* p_tc = new TwoCards();
            p_tc->mpFirstCard = p_card1;
            p_tc->mpSecondCard = p_card2;
            mStartingHands.Add(p_tc);
            //cout << p_card1->ToString() << " " << p_card2->ToString() << "\n";
        }
        cards.Clear();
    }
}

// class destructor
HoldemEmulator::~HoldemEmulator()
{
    // TODO: clear starting hands
}

/**
 * Calculates the probability of winning with the given cards
 * in preflop
 * @return winning probability in percents
 **/

int HoldemEmulator::PreFlopWinProbability(int numPlayers, const LinkedList<Card*>& rPocketCards, int numTests)
{

	Card* p_cards[7];

    Card* p_pocket_cards[2];
    p_pocket_cards[0] = rPocketCards[0];
    p_pocket_cards[1] = rPocketCards[1];

	CardStack card_stack;

	long wins = 0;
	
    //std::cout << "First poket card: ";
	//std::cin >> cardlabel;
	//pocketCards[0] = new Card(cardlabel);
	
	//std::cout << "Second poket card: ";
	//std::cin >> cardlabel;
	//pocketCards[1] = new Card(cardlabel);
	for (long i = 0; i < numTests; i++) {
		card_stack.Init();
		card_stack.TakeCard(p_pocket_cards[0]);
		card_stack.TakeCard(p_pocket_cards[1]);
		p_cards[0] = p_pocket_cards[0];
		p_cards[1] = p_pocket_cards[1];
		for (int j = 2; j < 7; j++) {
			p_cards[j] = card_stack.TakeRandomCard();
		}
		int strength = HoldemRules::GetHandStrength(p_cards, 7);
		bool does_win = true;
		for (int j = 0; j < numPlayers - 1; j++) {
			p_cards[0] = card_stack.TakeRandomCard();
			p_cards[1] = card_stack.TakeRandomCard();
			int base_strength = HoldemRules::GetHandStrength(p_cards, 7);
			if (strength <= base_strength) {
				does_win = false;
				break;
			}
		}
		if (does_win) {
            wins++;
        }
	}
	int winning_probability = (wins * 100 / numTests);
	//std::cout << "HoldemEmulator: numPlayers = " << numPlayers << " wins = " << wins * 100 / NUM_TESTS << "\n";
    //Debug(String("HoldemEmulator::PreFlop ") + (numPlayers) + " " + p_pocket_cards[0]->ToString() + " " + p_pocket_cards[1]->ToString() + " " + winning_probability);

    return winning_probability;
}

int HoldemEmulator::PostFlopWinProbability(int numPlayers, const LinkedList<Card*>& rPocketCards, const LinkedList<Card*>& rCommunityCards)
{
    Object::Dbg(String("PostFlopWinProbability 1"));
	Card* p_cards[7];

    Card* p_pocket_cards[2];
    p_pocket_cards[0] = rPocketCards[0];
    p_pocket_cards[1] = rPocketCards[1];

    int num_community_cards = rCommunityCards.Size();
    Card* p_community_cards[num_community_cards];
    
    for (int i = 0; i < num_community_cards; i++) {
        p_community_cards[i] = rCommunityCards[i];
    }

	CardStack card_stack;

	long wins = 0;
	
    //std::cout << "First poket card: ";
	//std::cin >> cardlabel;
	//pocketCards[0] = new Card(cardlabel);
	
	//std::cout << "Second poket card: ";
	//std::cin >> cardlabel;
	//pocketCards[1] = new Card(cardlabel);
	
    Object::Dbg(String("PostFlopWinProbability 2"));
	for (long i = 0; i < msNumTests; i++) {
		card_stack.Init();
		card_stack.TakeCard(p_pocket_cards[0]);
		card_stack.TakeCard(p_pocket_cards[1]);
		p_cards[0] = p_pocket_cards[0];
		p_cards[1] = p_pocket_cards[1];
		for (int j = 0; j < num_community_cards; j++) {
    		p_cards[j + 2] = p_community_cards[j];
    		card_stack.TakeCard(p_community_cards[j]);
        }
		for (int j = 2 + num_community_cards; j < 7; j++) {
		    Object::Dbg(String("PostFlopWinProbability 3.") + String(j));
			p_cards[j] = card_stack.TakeRandomCard();
		    Object::Dbg(String("PostFlopWinProbability 3a.") + String(j));
		}
		int strength = HoldemRules::GetHandStrength(p_cards, 7);
		bool does_win = true;
	    Object::Dbg(String("PostFlopWinProbability 4.") + String(i));
		for (int j = 0; j < numPlayers - 1; j++) {
			int rnd = rand() % mStartingHands.Size();
			Object::Dbg(String("PostFlopWinProbability 5: ") + String(rnd));
			for (int k = rnd; k < mStartingHands.Size(); k++) {
				if (card_stack.TakeCard(mStartingHands[k]->mpFirstCard)
					&& card_stack.TakeCard(mStartingHands[k]->mpSecondCard)) {
						p_cards[0] = mStartingHands[k]->mpFirstCard;
						p_cards[1] = mStartingHands[k]->mpSecondCard;
						break;
				}
			}
			if (!p_cards[0]) {
				for (int k = 0; k < rnd; k++) {
					if (card_stack.TakeCard(mStartingHands[k]->mpFirstCard)
						&& card_stack.TakeCard(mStartingHands[k]->mpSecondCard)) {
							p_cards[0] = mStartingHands[k]->mpFirstCard;
							p_cards[1] = mStartingHands[k]->mpSecondCard;
							break;
					}
				}
			}
			int base_strength = HoldemRules::GetHandStrength(p_cards, 7);
			if (strength <= base_strength) {
				does_win = false;
				break;
			}
		}
		if (does_win) {
            wins++;
        }
	}
    Object::Dbg(String("PostFlopWinProbability 4"));
	int winning_probability = (wins * 100 / msNumTests);
	//std::cout << "HoldemEmulator: numPlayers = " << numPlayers << " wins = " << wins * 100 / NUM_TESTS << "\n";
    //Debug(String("HoldemEmulator::PostFlop ") + (numPlayers) + " " + p_pocket_cards[0]->ToString() + " " + p_pocket_cards[1]->ToString() + " " + winning_probability);

    return winning_probability;
}
