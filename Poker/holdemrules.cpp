#include "holdemrules.h" // class's header file
#include "table.h" // class's header file

using namespace poker;

void HoldemRules::DealPocketCards()
{
    LinkedList<Player*>& r_players = mrTable.GetPlayers();

    for (LinkedList<Player*>::Iterator i = r_players.Begin(); !i.Done(); i++) {
        Debug("DealPocketCards 1");
        Card* p_card_1 = mrTable.GetCardStack().DealCard();
        Card* p_card_2 = mrTable.GetCardStack().DealCard();
        Debug("DealPocketCards 2");
        Debug(String("Pocket cards for [") + (*i)->Name() + "]: " + p_card_1->ToString() + " " + p_card_2->ToString());
        //if ((*i)->Name() == "You") {
        //    cout << "Pocket cards for [" << (*i)->Name() << "]: " << p_card_1->ToString() << " " << p_card_2->ToString() << "\n";
        //}
        Debug("DealPocketCards 3");
        (*i)->AddCard(p_card_1);
        //Debug("DealPocketCards 2");
        (*i)->AddCard(p_card_2);
        //Debug("DealPocketCards 3");
        Debug("DealPocketCards 4");
    }

    mrTable.RenderCards("Pocket cards");
}

void HoldemRules::DealFlop()
{
    Card* p_card_1 = mrTable.GetCardStack().DealCard();
    Card* p_card_2 = mrTable.GetCardStack().DealCard();
    Card* p_card_3 = mrTable.GetCardStack().DealCard();
    Debug(String("Flop: ") + p_card_1->ToString() + " " + p_card_2->ToString() + " " + p_card_3->ToString());

    mrTable.AddCommunityCard(p_card_1);
    mrTable.AddCommunityCard(p_card_2);
    mrTable.AddCommunityCard(p_card_3);

    mrTable.RenderCards("Flop");
}

void HoldemRules::DealTurn()
{
    Card* p_card_1 = mrTable.GetCardStack().DealCard();
    Debug(String("Turn: ") + p_card_1->ToString());
    mrTable.AddCommunityCard(p_card_1);
    mrTable.RenderCards("Turn");
}

void HoldemRules::DealRiver()
{
    Card* p_card_1 = mrTable.GetCardStack().DealCard();
    Debug(String("River: ") + p_card_1->ToString());
    mrTable.AddCommunityCard(p_card_1);
    mrTable.RenderCards("River");
}

void HoldemRules::DetermineWinners()
{
    //--------------------------------------------
    // Calculate best combinations for all players
    //Debug("DetermineWinners 1");
    const LinkedList<Card*>& r_comm_cards = mrTable.GetCommunityCards();
    //Debug("DetermineWinners 2");
    LinkedList<Player*>& players = mrTable.GetPlayers();
    //Debug("DetermineWinners 3");
    for (LinkedList<Player*>::Iterator i = players.Begin(); !i.Done(); i++) {
        if ((*i)->IsFolded()) {
    	    Debug(String("Player [") + (*i)->Name() + "] is folded");
            continue;
        }
        Debug(String("DetermineWinners 4: [") + (*i)->Name() + "]");
        const LinkedList<Card*>& r_player_cards = (*i)->GetCards();
        Debug("DetermineWinners 5");
        Card* cards[r_player_cards.Size() + r_comm_cards.Size()];
        Debug("DetermineWinners 6");
        //cout << "--------------------------------------\n";
        cout << '[' << (*i)->Name() << "] shows: ";
        int j;
        for (j = 0; j < r_player_cards.Size(); j++) {
            cards[j] = r_player_cards[j];
            Debug(r_player_cards[j]->ToString() + " ");
            cout << r_player_cards[j]->ToString() << ' ';
        }
        cout << '\n';
        Debug("--");
        for (j = r_player_cards.Size(); j < r_player_cards.Size() + r_comm_cards.Size(); j++) {
            cards[j] = r_comm_cards[j - r_player_cards.Size()];
            Debug(r_comm_cards[j - r_player_cards.Size()]->ToString() + " ");
        }
        Debug("\n");
        Debug("DetermineWinners 8");
	    (*i)->SetHandStrength(GetHandStrength(cards, r_player_cards.Size() + r_comm_cards.Size()));
	    Debug(String("Hand strength for [") + (*i)->Name() + "]: " + (long) (*i)->HandStrength());
        Debug("DetermineWinners 9");
    }
    
    //-------------------------------
    // Order players by hand strength
    
    LinkedList<LinkedList<Player*>* > winners;
    Debug("DetermineWinners 10");
    
    int num_winners = 0;
    for (;;) {
    //while (num_winners < players.Size()) {
        Debug("DetermineWinners 11");
        unsigned long max_strength = 0;
        for (LinkedList<Player*>::Iterator i = players.Begin(); !i.Done(); i++) {
            if ((*i)->IsFolded()) {
                continue;
            }
            if ((*i)->HandStrength() > max_strength) {
                max_strength = (*i)->HandStrength();
            }
        }
        if (max_strength <= 0) {
            Debug(String("DetermineWinners: Thats all"));
            // No more winners
            break;
        }
        Debug(String("DetermineWinners: max_strength = ") + (long) max_strength);
        LinkedList<Player*>* p_w = new LinkedList<Player*>();

        for (LinkedList<Player*>::Iterator i = players.Begin(); !i.Done(); i++) {
            if ((*i)->IsFolded()) {
                continue;
            }
            if ((*i)->HandStrength() == max_strength) {
                p_w->Add((*i));
                (*i)->SetHandStrength(0);
                num_winners++;
                Debug(String("DetermineWinners: adding winner: [") + (*i)->Name() + "]");
            }
        }
        Debug("DetermineWinners 111");
        if (p_w->Size() > 0) {
            Debug("DetermineWinners 1111");
            winners.Add(p_w);
            Debug("DetermineWinners 11111");
        }
    }

    Debug("DetermineWinners 12");
    //-----------------------------------------------------    
    // Iterate ower winners and add wins to player balances
    
    for (LinkedList<LinkedList<Player*>* >::Iterator i = winners.Begin(); !i.Done(); i++) {
        mrTable.GetPotManager().Win(*(*i));
        delete (*i);
    }
    winners.Clear();
}
