#include <stdlib.h>
#include "base/string.h"
#include "base/thread.h"
#include "base/linkedlist.h"
#include "table.h"
#include "seat.h"
#include "player.h"
#include "action.h"
#include "holdememulator.h"

using namespace poker;

HoldemEmulator he;

int GetAction(const Table& rTable, const Player& rPlayer, const LinkedList<Action*>& rAvailableActions)
{
    Object::Dbg(String("GetAction 1"));
    int win_probability;
    const LinkedList<Card*>& r_community_cards = rTable.GetCommunityCards();
    Object::Dbg(String("GetAction 2"));
    const LinkedList<Card*>& r_pocket_cards = rPlayer.GetCards();
    Object::Dbg(String("GetAction 3"));
    if (r_community_cards.Size() <= 0) {
        win_probability = he.PreFlopWinProbability(rTable.CountPlayers(), r_pocket_cards);
    }
    else {
        win_probability = he.PostFlopWinProbability(rTable.CountPlayers(), r_pocket_cards, r_community_cards);
    }
    int pot = rTable.TotalBet();
    //-------------------------
    //if (rTable.CurrentBet() > rTable.BetValue()) {
    //    pot += rTable.CountPlayers() * (4 * rTable.BetValue() - rTable.CurrentBet());
    //}
    Object::Dbg(String("GetAction 4"));
    if (pot < rTable.CountPlayers() * rTable.BetValue() 
        && r_community_cards.Size() <= 0) {
        pot = rTable.CountPlayers() * rTable.BetValue();
    }
    //else {
    //    pot += rTable.CountPlayers() * rTable.BetValue();
    //}
    Object::Dbg(String("GetAction 5"));
    // Add random factor (looseness, tightness)
    int rnd = (rand() % 40) - 20;
    pot += pot * rPlayer.GetTightness() / 100;
    
    //-------------------------
    int player_bet = rPlayer.TotalBet();
    int win = pot * win_probability / 100;
    int max_allowed_bet = win - player_bet;
    if (r_community_cards.Size() > 0) {
        max_allowed_bet *= 2;
    }
    if (max_allowed_bet < 0) {
        max_allowed_bet = 0;
    }
    Object::Dbg(String("estimated_pot: ") + pot + ", player_bet: " + player_bet + ", win: " + win);
    Object::Dbg(String("max_allowed_bet: ") + max_allowed_bet);
    int index;
    int min_max_sum = 0;
    int i = rAvailableActions.Size() - 1;
    for (LinkedList<Action*>::Iterator i_action = rAvailableActions.End(); !i_action.Done(); i_action--) {
        if ((*i_action)->Sum() >= min_max_sum && (*i_action)->Sum() <= max_allowed_bet) {
            index = i;
            min_max_sum = (*i_action)->Sum();
        }
        i--;
    }
    Object::Dbg(String("allowed_action: ") + rAvailableActions[index]->ToString());
    return index + 1;
}

int main(char* args[])
{

    //------------------------------------------------
    //Card* cards[7];
    
    //cards[0] = new Card("CK");
    //cards[1] = new Card("SQ");
    
    //cards[2] = new Card("CK");
    //cards[3] = new Card("H3");
    //cards[4] = new Card("ST");
    //cards[5] = new Card("H7");
    //cards[6] = new Card("HT");
    //long p1 = HoldemRules::GetHandStrength(cards, 2);
    //cards[0] = new Card("D7");
    //cards[1] = new Card("C7");
    //long p2 = HoldemRules::GetHandStrength(cards, 7);
    //cout << p1 << "\n";
    //cout << p2 << "\n";
    
    //int i;
    //cin >> i;
    //------------------------------------------------
    try {
    	Mutex mutex;
        Table table(&mutex);
        
        for (int i = 0; i < 9; i++) {
            new Player(String("Bot ") + (i + 1), *table.FirstFreeSeat(), 1000);
        }
        new Player(String("You"), *table.FirstFreeSeat(), 1000);
        
        // Start the table thread
        table.Start();

        for (;;) {
        	SynchBlock synch_block = mutex.GetSynchBlock();
        	synch_block.Begin();
            Object::Dbg(String("main 1"));
			Player* p_player = table.GetActivePlayer();
			if (p_player) {
				Object::Dbg(String("main 3"));
				Object::Dbg(String("Active player: ") + p_player->ToString());
				//cout << String("Active player: ") + (*i_player)->ToString() + "\n";
				const LinkedList<Action*>& r_available_actions = p_player->GetAvailableActions();
				int index = 1;
				int option = 1;
				if (r_available_actions.Size() > 1) {
					Object::Dbg(String("main 4"));
					if (p_player->Name() == "You") {
						Object::Dbg(String("main 5"));
						do {
							cout << "Pot: " << table.TotalBet() << '\n';
							for (LinkedList<Action*>::Iterator i_action = r_available_actions.Begin(); !i_action.Done(); i_action++) {
								cout << (String() + (index++) + ": " + (*i_action)->ToString() + "\n");
							}
							cout << "Your bet please: ";
							cin >> option;
						} while (option < 1 || option > r_available_actions.Size());
					}
					else {
						Object::Dbg(String("main 6"));
						option = GetAction(table, *p_player, r_available_actions);
					}
				}
				Object::Dbg("main 7");
				Object::Dbg(r_available_actions[option - 1]->ToString());
				table.TakeAction(*p_player, *r_available_actions[option - 1]);
				Object::Dbg(String("main 8"));
            }
        	synch_block.End();
            Thread::Sleep(100);
        }
    }
    catch (...) {
        cout << "ERROR!\n";
        cin;
    }
}
