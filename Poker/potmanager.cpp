#include "potmanager.h" // class's header file

using namespace poker;

void PotManager::CollectChips(const LinkedList<Player*>& rPlayers)
{
    if (mPots.Size() <= 0) {
        Debug("CollectChips: No pots yet, creating one");
        mPots.Add(new Pot());
    }
    int player_bets[rPlayers.Size()];
    int i = 0;
    for (LinkedList<Player*>::Iterator i_player = rPlayers.Begin(); !i_player.Done(); i_player++) {
        player_bets[i++] = (*i_player)->CurrentBet();
        //Debug(String("CollectChips: player_bets[") + i + "]= " + (*i_player)->CurrentBet());
    }
    bool first_time = true;
    for (;;) {
        int min_bet = -1;
        for (int i = 0; i < rPlayers.Size(); i++) {
            if (player_bets[i] > 0 && (min_bet < 0 || player_bets[i] < min_bet)) {
                min_bet = player_bets[i];
                //Debug(String("CollectChips: minbet=") + min_bet);
            }
        }
        Debug(String("CollectChips: min_bet = ") + min_bet);
        if (min_bet <= 0) {
            return;
        }
        if (!first_time) {
            Debug(String("CollectChips: creating sidepot"));
            mPots.Add(new Pot());
        }
        Pot* p_current_pot = mPots[mPots.Size() - 1];
        int i = 0;
        for (LinkedList<Player*>::Iterator i_player = rPlayers.Begin(); !i_player.Done(); i_player++) {
            if (player_bets[i] >= min_bet) {
                player_bets[i] -= min_bet;
                if (!(*i_player)->IsFolded()) {
                    p_current_pot->Add(*(*i_player), min_bet);
                }
                else {
                    p_current_pot->Add(min_bet);
                }
            }
            i++;
        }
        first_time = false;
    }
    cout << "Pots: ";
    for (LinkedList<Pot*>::Iterator i_pot = mPots.Begin(); !i_pot.Done(); i_pot++) {
         cout << (*i_pot)->ToString() << " ";
    }
    cout << '\n';
}

bool PotManager::Win(LinkedList<Player*>& rPlayers)
{
    int num_empty = 0;
    for (LinkedList<Pot*>::Iterator i_pot = mPots.Begin(); !i_pot.Done(); i_pot++) {
        int num_players = 0;
        for (LinkedList<Player*>::Iterator i_player = rPlayers.Begin(); !i_player.Done(); i_player++) {
            if ((*i_pot)->Contains(*i_player)) {
                num_players++;
            }
        }
        for (LinkedList<Player*>::Iterator i_player = rPlayers.Begin(); !i_player.Done(); i_player++) {
            if ((*i_pot)->Contains(*i_player)) {
                int win_amount = (*i_pot)->TotalAmount();
                if (num_players > 1) {
                    win_amount /= num_players;
                }
                (*i_player)->Win(win_amount);
                (*i_pot)->mTotalAmount -= win_amount;
                num_players--;
            }
        }
        if ((*i_pot)->TotalAmount() <= 0) {
            num_empty++;
        }
    }
    return num_empty >= mPots.Size();
}

String PotManager::ToString() const
{
    String ret_val;
    for (LinkedList<Pot*>::Iterator i_pot = mPots.Begin(); !i_pot.Done(); i_pot++) {
        ret_val = ret_val + "    " + (*i_pot)->ToString() + "\n";
    }
    return ret_val;
}

void PotManager::ClearPots()
{
    for (LinkedList<Pot*>::Iterator i_pot = mPots.Begin(); !i_pot.Done(); i_pot++) {
        delete *i_pot;
    }
    mPots.Clear();
}

// Pot ---------------------------------------------------------

void PotManager::Pot::Add(Player& rPlayer, int amount)
{
    if (amount <= 0) {
        return;
    }
    if (!Contains(&rPlayer)) {
        LinkedList<Player*>::Add(&rPlayer);
    }
    mTotalAmount += amount;
}

void PotManager::Pot::Add(int amount)
{
    if (amount <= 0) {
        return;
    }
    mTotalAmount += amount;
}

String PotManager::Pot::ToString() const
{
    String ret_val(mTotalAmount);
    for (LinkedList<Player*>::Iterator i_player = Begin(); !i_player.Done(); i_player++) {
        ret_val = ret_val + ", " + (*i_player)->Name();
    }
    return ret_val;
}
