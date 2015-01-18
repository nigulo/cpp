#include "action.h" // class's header file
#include "player.h"
#include "seat.h"
#include "table.h"

using namespace poker;

const Action::Type Action::Type::msSmallBlind(String("SB"));
const Action::Type Action::Type::msBigBlind(String("BB"));
const Action::Type Action::Type::msCheck(String("Check"));
const Action::Type Action::Type::msBet(String("Bet"));
const Action::Type Action::Type::msCall(String("Call"));
const Action::Type Action::Type::msRaise(String("Raise"));
const Action::Type Action::Type::msFold(String("Fold"));
const Action::Type Action::Type::msAllin(String("Allin"));

// class constructor
Action::Action(Player& rPlayer, const Type& rType, int sum) :
    mrPlayer(rPlayer),
    mrType(rType),
    mSum(sum),
    mCurrentPlayerBet(rPlayer.CurrentBet()),
    mCurrentBet(rPlayer.GetSeat().GetTable().CurrentBet())
{
	mpPreviousAction = rPlayer.GetSeat().GetTable().GetLastAction();
	mpPreviousPlayerAction = rPlayer.GetLastAction();
}

// class destructor
Action::~Action()
{
	// insert your code here
}

String Action::ToString() const
{
    return mrType.Value() + " " + mSum;
}
