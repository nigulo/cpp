#include "player.h" // class's header file
#include "seat.h"
#include "table.h"

using namespace poker;

Player::Player(const String& rName, Seat& rSeat, int initalChips) :
    mName(rName),
    mrSeat(rSeat),
    mChips(initalChips),
    mTotalBet(0),
    mCurrentBet(0),
    mWin(0),
    mActive(false),
    mHandStrength(0),
    mSitIn(true),
    mTightness(50)
{
    rSeat.SitDown(*this);
}

Player::~Player()
{
    mrSeat.StandUp(*this);
}

void Player::BuyChips(int amount)
{
    mChips += amount;
}

void Player::GameStarted()
{
    mHandStrength = 0;
    mCurrentBet = 0;
    mTotalBet = 0;
    mWin = 0;
    mCards.Clear();
    for (LinkedList<Action*>::Iterator i = mActions.Begin(); !i.Done(); i++) {
        delete (*i);
    }
    mActions.Clear();
    if (mName == "You") {
        cout << (String("[") + Name() + "] chips: " + mChips + "\n");
    }

    //mTightness = (rand() % 100) - 50;
}

void Player::RoundFinished()
{
    mCurrentBet = 0;
    for (LinkedList<Action*>::Iterator i = mRoundActions.Begin(); !i.Done(); i++) {
        delete (*i);
    }
    mRoundActions.Clear();
}

void Player::AskAction()
{
    //----------------------------------
    // Set tightness
    int relative_pos = mrSeat.GetTable().GetDealer()->GetSeat().Number();
    if (relative_pos <= mrSeat.Number()) {
        relative_pos += 10;
    }
    relative_pos -= mrSeat.Number();
    mTightness = relative_pos * 5 - 25;
    //----------------------------------
    mActive = true;
}

void Player::TakeAction(Action& rAction) 
{
    if (mAvailableActions.IndexOf(&rAction) < 0) {
        throw String("Illegal action");
    }
    mActions.Add(new Action(rAction));
    mRoundActions.Add(new Action(rAction));
    mTotalBet += rAction.Sum();
    mChips -= rAction.Sum();
    //cout << "--------------------------------------\n";

    cout << (String("[") + Name() + "]: " + rAction.ToString() + "\n");

    Debug(ToString() + ": " + rAction.ToString());
    Debug(String("currentbet = ") + mCurrentBet);
    mCurrentBet += rAction.Sum();
    for (LinkedList<Action*>::Iterator i = mAvailableActions.Begin(); !i.Done(); i++) {
        delete (*i);
    }
    mAvailableActions.Clear();
    mActive = false;
}

void Player::AddAvailableAction(const Action& rAction) 
{
    mAvailableActions.Add(new Action(rAction));
}

Action* Player::GetLastAction() const
{
    if (mActions.Size() > 0) {
        //Debug(String("GetLastAction: ") + (*(mActions.End()))->ToString());
        return *(mActions.End());
    }
    return 0;
}

Action* Player::GetLastRoundAction() const
{
    if (mRoundActions.Size() > 0) {
        //Debug(String("GetLastRoundAction: ") + (*(mRoundActions.End()))->ToString());
        return *(mRoundActions.End());
    }
    return 0;
}

void Player::AddCard(Card* pCard)
{
    mCards.Add(pCard);
}

bool Player::IsAllin() const
{
    return GetLastAction() && &GetLastAction()->GetType() == &Action::Type::msAllin;
}

bool Player::IsFolded() const
{
    return GetLastAction() && &GetLastAction()->GetType() == &Action::Type::msFold;
}

void Player::Win(int amount) 
{
    if (amount > 0) {
        mWin += amount;
        mChips += amount;
        //cout << "--------------------------------------\n";
        cout << (String("[") + Name() + "] wins " + amount + "\n");
    
        Debug(ToString() + " wins " + amount);
    }
}

String Player::ToString() const
{
    return
    String("name=") + mName + 
    String(",seat=") + mrSeat.ToString() + 
    String(",chips=") + mChips +
    String(",current_bet=") + mCurrentBet + 
    String(",total_bet=") + mTotalBet + 
    String(",win=") + mWin;
}
