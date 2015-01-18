#include <typeinfo>

#include "table.h" // class's header file
#include "holdemrules.h" // class's header file

using namespace poker;

const Table::GameStage::Type Table::GameStage::Type::msStart;
const Table::GameStage::Type Table::GameStage::Type::msFlop;
const Table::GameStage::Type Table::GameStage::Type::msTurn;
const Table::GameStage::Type Table::GameStage::Type::msRiver;
const Table::GameStage::Type Table::GameStage::Type::msFinish;
const Table::GameStage::Type Table::GameStage::Type::msPreFlop;


// class constructor
Table::Table(Mutex* pMutex) :
    mCurrentBet(0),
    mpGameRules(new HoldemRules(*this)),
    mpCurrentPlayer(0),
    mpDealer(0),
    mNumRaises(0),
    mpMutex(pMutex)
{
    mpGameStage = 0;
    //mpSeats = new Seat*[msNumSeats];
    for (int i = 0; i < msNumSeats; i++) {
        mpSeats[i] = new Seat(*this, i);
    }
}

// class destructor
Table::~Table()
{
    for (int i = 0; i < msNumSeats; i++) {
        delete mpSeats[i];
    }
}

String Table::ToString() const
{
    Debug("1");
    String ret_val;
    if (mpDealer) {
        ret_val = ret_val + "dealer: " + mpDealer->ToString() + "\n";
    }
    Debug("2");
    if (mpCurrentPlayer) {
        ret_val = ret_val + "current_player: " + mpCurrentPlayer->ToString() + "\n";
    }
    Debug("3");
    ret_val = ret_val + "current_bet: " + mCurrentBet + "\n";
    ret_val = ret_val + "players:\n";
    Debug("4");

    for (LinkedList<Player*>::Iterator i_player = mPlayers.Begin(); !i_player.Done(); i_player++) {
        ret_val = ret_val + "    " + (*i_player)->ToString() + "\n";
    }
    Debug("5");
    ret_val = ret_val + "community_cards:\n";
    for (LinkedList<Card*>::Iterator i_card = mCommunityCards.Begin(); !i_card.Done(); i_card++) {
        ret_val = ret_val + "    " + (*i_card)->ToString() + "\n";
    }
    Debug("6");
    ret_val = ret_val + "pots:\n    " + mPotManager.ToString() + "\n";
    Debug("7");
    
    //for (int i = 0; i < msNumSeats; i++) {
    //    retVal += mpSeats[i]->ToString();
    //}
    return ret_val;
}

void Table::Run()
{
    try {
        while (true) {
        	SynchBlock synch_block = mpMutex->GetSynchBlock();
        	synch_block.Begin();
            Play();
            synch_block.End();
            Thread::Sleep(100);
        }
    }
    catch (String ex) {
        cout << ex;
        Debug(ex);
    }
}

Action* Table::GetLastAction() const
{
    if (mActions.Size() > 0) {
        return *(mActions.End());
    }
    return 0;
}

Action* Table::GetLastRoundAction() const
{
    if (mRoundActions.Size() > 0) {
        return *(mRoundActions.End());
    }
    return 0;
}

void Table::Play()
{
    Debug("Play 1");
    if (mpCurrentPlayer && mpCurrentPlayer -> IsActive()) {
    	return;
    }
    if (!mpGameStage) {
        mpGameStage = new GameStage(GameStage::Type::msStart);
    }
    if (&(mpGameStage->mrType) == &(GameStage::Type::msStart)) {
        StartGame();
    }
    if (typeid(*mpGameStage) == typeid(PreFlopStage)) {
        PreFlop();
    }
    else if (&(mpGameStage->mrType) == &(GameStage::Type::msFlop)) {
        Flop();
    }
    else if (&(mpGameStage->mrType) == &(GameStage::Type::msTurn)) {
        Turn();
    }
    else if (&(mpGameStage->mrType) == &(GameStage::Type::msRiver)) {
        River();
    }
    else if (&(mpGameStage->mrType) == &(GameStage::Type::msFinish)) {
        Finish();
        return;
    }
    Debug(ToString());
    if (mpCurrentPlayer) {
        mpCurrentPlayer->AskAction();
    }
    else {
        mpCurrentPlayer = mpDealer;//GetNextPlayer(*mpDealer);
    }
}

void Table::StartGame()
{
    cout << "============ Game started ============\n";
    Debug("StartGame 1");
    //----------------------
    // Initialize game state
    mPlayers.Clear();

    Debug("StartGame 2");
    
    mCommunityCards.Clear();
    
    Debug("StartGame 4");
    //mPotManager.ClearPots();
    
    for (LinkedList<Action*>::Iterator i = mActions.Begin(); !i.Done(); i++) {
        delete (*i);
    }
    mActions.Clear();
    Debug("StartGame 41");

    mPotManager.ClearPots();
    //for (LinkedList<Action*>::Iterator i = mRoundActions.Begin(); !i.Done(); i++) {
    //    delete (*i);
    //}
    //mRoundActions.Clear();
    
    Debug("StartGame 5");
    
    Debug("StartGame 6");
    mCurrentBet = 0;
    mNumRaises = 0;
    //----------------------
    
    Debug("StartGame 7");
    for (int i = 0; i < msNumSeats; i++) {
        Player* p_player = mpSeats[i]->GetPlayer();
        if (!p_player || !p_player->IsSitIn()) {
            continue;
        }
        Debug("StartGame 71");
        p_player->GameStarted();
        Debug("StartGame 72");
        mPlayers.Add(p_player);
        Debug("StartGame 73");
    }
    Debug("StartGame 8");
    mCards.Shuffle();
    mpGameStage = new PreFlopStage();
    Debug("StartGame 9");
}

void Table::PreFlop()
{
    Debug("PreFlop 1");
    PreFlopStage* p_game_stage = static_cast<PreFlopStage*>(mpGameStage);
    if (!p_game_stage->mDealerMoved) {
        Debug("PreFlop 2");
        MoveDealer();
        Debug("PreFlop 3");
    }
    Debug("PreFlop 4");
    if (!p_game_stage->mSmallBlindPosted || !p_game_stage->mBigBlindPosted) {
        Debug("PreFlop 5");
        Blinds();
        Debug("PreFlop 6");
    }
    else if (!mpGameStage->mCardsDealt) {
        Debug("PreFlop 7");
        mpGameRules->DealPocketCards();
        Debug("PreFlop 8");
        mpGameStage->mCardsDealt = true;
        Debug("PreFlop 9");
        BettingRound();
    }
    else {
        Debug("PreFlop 10");
        BettingRound();
        Debug("PreFlop 11");
    }
}

void Table::Flop()
{
    Debug("Flop 1");
    if (!mpGameStage->mCardsDealt) {
        Debug("Flop 2");
        mpGameRules->DealFlop();
        Debug("Flop 3");
        mpGameStage->mCardsDealt = true;
        Debug("Flop 4");
    }
    Debug("Flop 5");
    BettingRound();
    Debug("Flop 6");
}

void Table::Turn()
{
    Debug("Turn 1");
    if (!mpGameStage->mCardsDealt) {
        Debug("Turn 2");
        mpGameRules->DealTurn();
        Debug("Turn 3");
        mpGameStage->mCardsDealt = true;
        Debug("Turn 4");
    }
    Debug("Turn 5");
    BettingRound();
    Debug("Turn 6");
}

void Table::River()
{
    Debug("River 1");
    if (!mpGameStage->mCardsDealt) {
        Debug("River 2");
        mpGameRules->DealRiver();
        Debug("River 3");
        mpGameStage->mCardsDealt = true;
        Debug("River 4");
    }
    Debug("River 5");
    BettingRound();
    Debug("River 6");
}

void Table::Finish()
{
    Debug("Finish 1");
    mpGameRules->DetermineWinners();
    Debug("Finish 2");
    GameStage* p_stage = mpGameStage;
    mpGameStage = 0;
    delete p_stage;
}

void Table::BettingRound()
{
    Debug("BettingRound 1");
    if (RoundFinished()) {
        Debug("BettingRound 2");
        mPotManager.CollectChips(mPlayers);
        Debug("BettingRound 3");
        for (LinkedList<Player*>::Iterator i_player = mPlayers.Begin(); !i_player.Done(); i_player++) {
            (*i_player)->RoundFinished();
        }
        Debug("BettingRound 4");
        mpCurrentPlayer = nullptr;
        mCurrentBet = 0;
        mNumRaises = 0;
        for (LinkedList<Action*>::Iterator i_action = mRoundActions.Begin(); !i_action.Done(); i_action++) {
            delete (*i_action);
        }
        mRoundActions.Clear();
        NextRound();
        Debug("BettingRound 5");
        return;
    }
    
    Debug("BettingRound 6");
    Player* p_next_player = GetNextPlayer();
    if (!p_next_player) {
        throw String("Cannot continue betting round: no next player found");
    }
    Debug(String("BettingRound next_player: ") + p_next_player->ToString());

    int available_bet = BetValue();
    
    int bet_amount = available_bet;
    int underraise_amount = mCurrentBet % available_bet;
    int call_amount = (mCurrentBet - underraise_amount) - p_next_player->CurrentBet();
    int raise_amount = call_amount + available_bet;

    bool check_available = true;
    bool bet_available = true;
    bool call_available = true;
    bool raise_available = mNumRaises < 3 && mCurrentBet > 0;
    bool allin_available = false;

    if (mCurrentBet > 0) {
        bet_available = false;
        if (p_next_player->CurrentBet() != mCurrentBet) {
            check_available = false;
        }
    }
    if (p_next_player->Chips() < bet_amount && bet_available) {
        bet_available = false;
        allin_available = true;
    }
    if (call_amount <= 0) {
        call_available = false;
    }
    if (p_next_player->Chips() < call_amount && call_available) {
        call_available = false;
        raise_available = false;
        allin_available = true;
    }
    if (p_next_player->Chips() < raise_amount && raise_available) {
        raise_available = false;
        allin_available = true;
    }
    int allin_amount = p_next_player->Chips();
    Debug(String("BettingRound: check_available=") + check_available);
    Debug(String("BettingRound: bet_available=") + bet_available + ", bet_amount=" + bet_amount);
    Debug(String("BettingRound: call_available=") + call_available + ", call_amount=" + call_amount);
    Debug(String("BettingRound: raise_available=") + raise_available + ", raise_amount=" + raise_amount);
    Debug(String("BettingRound: allin_available=") + allin_available + ", allin_amount=" + allin_amount);
    
    if (check_available) {
        p_next_player->AddAvailableAction(Action(*p_next_player, Action::Type::msCheck, 0));
    }
    if (bet_available) {
        p_next_player->AddAvailableAction(Action(*p_next_player, Action::Type::msBet, bet_amount));
    }
    if (call_available) {
        p_next_player->AddAvailableAction(Action(*p_next_player, Action::Type::msCall, call_amount));
    }
    if (raise_available) {
        p_next_player->AddAvailableAction(Action(*p_next_player, Action::Type::msRaise, raise_amount));
    }
    p_next_player->AddAvailableAction(Action(*p_next_player, Action::Type::msFold, 0));

    if (allin_available) {
        p_next_player->AddAvailableAction(Action(*p_next_player, Action::Type::msAllin, allin_amount));
    }


    mpCurrentPlayer = p_next_player;
}

bool Table::RoundFinished() const
{
    if (GameFinished()) {
        return true;
    }
    Debug("RoundFinished 1");
    for (LinkedList<Player*>::Iterator i_player = mPlayers.Begin(); !i_player.Done(); i_player++) {
        //Debug(String("RoundFinished 2: lastaction: ") + ((*i_player)->GetLastAction() != 0));
        if (!(*i_player)->IsAllin() && 
            !(*i_player)->IsFolded() && 
            (!(*i_player)->GetLastRoundAction() ||
            &(*i_player)->GetLastRoundAction()->GetType() == &Action::Type::msBigBlind ||
            mCurrentBet > (*i_player)->CurrentBet())
            ) {
            Debug("RoundFinished 2");
            return false;
        }
    }
    Debug("RoundFinished 3");
    return true;
}

bool Table::GameFinished() const
{
    Debug("GameFinished 1");
    int num_players = 0;
    for (LinkedList<Player*>::Iterator i_player = mPlayers.Begin(); !i_player.Done(); i_player++) {
        //Debug(String("GameFinished player: ") + (*i_player)->ToString());
        //Debug(String("GameFinished player isAllin: ") + (*i_player)->IsAllin());
        //Debug(String("GameFinished player isFolded: ") + (*i_player)->IsFolded());
        if (//!(*i_player)->IsAllin() && 
            !(*i_player)->IsFolded()
            ) {
            Debug("GameFinished 2");
            num_players++;
        }
    }
    return num_players <= 1;
}

void Table::NextRound()
{
    if (GameFinished() || &mpGameStage->mrType == &GameStage::Type::msRiver) {
        GameStage* p_stage = mpGameStage;
        mpGameStage = new GameStage(GameStage::Type::msFinish);
        delete p_stage;
        Debug("NextRound: -> finish");
    }
    else if (typeid(*mpGameStage) == typeid(PreFlopStage)) {
        PreFlopStage* p_stage = static_cast<PreFlopStage*>(mpGameStage);
        mpGameStage = new GameStage(GameStage::Type::msFlop);
        delete p_stage;
        Debug("NextRound: preflop -> flop");
    }
    else if (&mpGameStage->mrType == &GameStage::Type::msFlop) {
        GameStage* p_stage = mpGameStage;
        mpGameStage = new GameStage(GameStage::Type::msTurn);
        delete p_stage;
        Debug("NextRound: flop -> turn");
    }
    else if (&mpGameStage->mrType == &GameStage::Type::msTurn) {
        GameStage* p_stage = mpGameStage;
        mpGameStage = new GameStage(GameStage::Type::msRiver);
        delete p_stage;
        Debug("NextRound: turn -> river");
    }
}

void Table::MoveDealer()
{
    Debug("MoveDealer 1");
    int index = 0;
    if (mpDealer) {
        Debug(String("MoveDealer previous_dealer: ") + mpDealer->ToString());
        index = mpDealer->GetSeat().Number() + 1;
    }
    Player* p_next_player = GetNextPlayer(true, index);
    if (!p_next_player) {
        throw String("Cannot start game: no dealer available");
    }
    mpDealer = p_next_player;
    Debug(String("MoveDealer next_dealer: ") + mpDealer->ToString());
    mpCurrentPlayer = p_next_player;
    PreFlopStage* p_game_stage = static_cast<PreFlopStage*>(mpGameStage);
    p_game_stage->mDealerMoved = true;
}

void Table::Blinds()
{
    Debug("Blinds 1");
    PreFlopStage* p_game_stage = static_cast<PreFlopStage*>(mpGameStage);
    if (!p_game_stage->mSmallBlindPosted) {
        Debug("Blinds 2");
        Player* p_sb_player = GetNextPlayer();
        if (!p_sb_player) {
            throw String("Cannot continue game: no SB player available");
        }
        p_sb_player->AddAvailableAction(Action(*p_sb_player, Action::Type::msSmallBlind, msSmallBlind));
        Debug(String("SB player: ") + p_sb_player->ToString());
        mpCurrentPlayer = p_sb_player;
        p_game_stage->mSmallBlindPosted = true;
    }
    else if (!p_game_stage->mBigBlindPosted) {
        Debug("Blinds 3");
        Player* p_bb_player = GetNextPlayer();
        if (!p_bb_player) {
            throw String("Cannot continue game: no BB player available");
        }
        p_bb_player->AddAvailableAction(Action(*p_bb_player, Action::Type::msBigBlind, msBigBlind));
        mpCurrentPlayer = p_bb_player;
        Debug(String("BB player: ") + p_bb_player->ToString());
        p_game_stage->mBigBlindPosted = true;
    }
}

Seat* Table::GetNextSeat(bool taken, int index) const
{
    for (int i = index; i < msNumSeats; i++) {
        if ((mpSeats[i]->GetPlayer() && taken) || 
            (!mpSeats[i]->GetPlayer() && !taken)) {
            return mpSeats[i];
        }
    }
    for (int i = 0; i < index; i++) {
        if ((mpSeats[i]->GetPlayer() && taken) || 
            (!mpSeats[i]->GetPlayer() && !taken)) {
            return mpSeats[i];
        }
    }
    return 0;
}

Player* Table::GetNextPlayer(bool sitIn, int index, bool allin, bool folded) const
{
    for (int i = index; i < msNumSeats; i++) {
        if (mpSeats[i]->GetPlayer() && 
            mpSeats[i]->GetPlayer()->IsSitIn() == sitIn && 
            mpSeats[i]->GetPlayer()->IsAllin() == allin && 
            mpSeats[i]->GetPlayer()->IsFolded() == folded
            ) {
            return mpSeats[i]->GetPlayer();
        }
    }
    for (int i = 0; i < index; i++) {
        if (mpSeats[i]->GetPlayer() && 
            mpSeats[i]->GetPlayer()->IsSitIn() == sitIn &&
            mpSeats[i]->GetPlayer()->IsAllin() == allin && 
            mpSeats[i]->GetPlayer()->IsFolded() == folded
            ) {
            return mpSeats[i]->GetPlayer();
        }
    }
    return 0;
}

Player* Table::GetNextPlayer() const
{
    if (!mpCurrentPlayer) {
        throw String("Cannot get next player: no current player");
    }
    return GetNextPlayer(true, mpCurrentPlayer->GetSeat().Number() + 1);
}

Player* Table::GetActivePlayer()
{
	Player* p_active_player = nullptr;
    LinkedList<Player*>& r_players = GetPlayers();
    Object::Dbg(String("main 1"));
    for (LinkedList<Player*>::Iterator i_player = r_players.Begin(); !i_player.Done(); i_player++) {
        Object::Dbg(String("main 2"));
        if ((*i_player)->IsActive()) {
        	p_active_player = *i_player;
        	break;
        }
    }
	return p_active_player;
}

Player* Table::GetNextPlayer(const Player& rPlayer) const
{
    return GetNextPlayer(true, rPlayer.GetSeat().Number() + 1);
}

Seat* Table::FirstFreeSeat() const
{
    return GetNextSeat(false);
}
//Player* Table::GetNextPlayer(const Player& rPlayer) const
//{
//    return GetNextPlayer(true, rPlayer.GetSeat().Number());
//}

void Table::TakeAction(Player& rPlayer, Action& rAction) {
	rPlayer.TakeAction(rAction);
	MakeAction(rAction);
}

void Table::MakeAction(Action& rAction)
{
    if (&rAction.GetType() == &Action::Type::msSmallBlind) {
        PreFlopStage* p_game_stage = static_cast<PreFlopStage*>(mpGameStage);
        p_game_stage->mSmallBlindPosted = true;
    }
    else if (&rAction.GetType() == &Action::Type::msBigBlind) {
        PreFlopStage* p_game_stage = static_cast<PreFlopStage*>(mpGameStage);
        p_game_stage->mBigBlindPosted = true;
    }
    else if (&rAction.GetType() == &Action::Type::msBet) {
    }
    else if (&rAction.GetType() == &Action::Type::msCall) {
    }
    else if (&rAction.GetType() == &Action::Type::msCheck) {
    }
    else if (&rAction.GetType() == &Action::Type::msFold) {
    }
    else if (&rAction.GetType() == &Action::Type::msRaise) {
    }
    else if (&rAction.GetType() == &Action::Type::msAllin) {
    }
    int bet_amount = rAction.Sum() + rAction.CurrentPlayerBet();
    Action* p_last_action = GetLastRoundAction();
    if (p_last_action 
        && &p_last_action->GetType() != &Action::Type::msCheck
        && &p_last_action->GetType() != &Action::Type::msSmallBlind
        && bet_amount >= mCurrentBet + BetValue() / 2) {
            
        mNumRaises++;
    }
    if (bet_amount > mCurrentBet) {
        mCurrentBet = bet_amount;
    }
    mActions.Add(new Action(rAction));
    mRoundActions.Add(new Action(rAction));
}

void Table::AddCommunityCard(Card* pCard)
{
    mCommunityCards.Add(pCard);
}

int Table::BetValue() const
{
    int available_bet = msBigBlind;
    if (&mpGameStage->mrType == &(GameStage::Type::msTurn) ||
        &mpGameStage->mrType == &(GameStage::Type::msRiver)) {
            available_bet += msBigBlind;
    }
    return available_bet;
}

int Table::TotalBet() const
{
    int total_bet = 0;
    for (LinkedList<Player*>::Iterator i_player = mPlayers.Begin(); !i_player.Done(); i_player++) {
        total_bet += (*i_player)->TotalBet();
    }
    return total_bet;
}

void Table::RenderCards(const String& rText) const
{
    //------------------------------------------------
    cout << "**************************************\n";
    const LinkedList<Card*>& rCards = GetCommunityCards();
    cout << rText << ": ";
    if (rCards.Size() > 0) {
        for (LinkedList<Card*>::Iterator i_card = rCards.Begin(); !i_card.Done(); i_card++) {
            cout << (*i_card)->ToString() << " ";
        }
        cout << "\n";
    }
    cout << "[You]: ";
    for (LinkedList<Player*>::Iterator i_player = mPlayers.Begin(); !i_player.Done(); i_player++) {
        if ((*i_player)->Name() == "You") {
            const LinkedList<Card*>& rPocketCards = (*i_player)->GetCards();
            for (LinkedList<Card*>::Iterator i_card = rPocketCards.Begin(); !i_card.Done(); i_card++) {
                cout << (*i_card)->ToString() << " ";
            }
        }
    }
    cout << "\n";
}

int Table::CountPlayers() const
{
    int num_players = 0;
    for (LinkedList<Player*>::Iterator i_player = mPlayers.Begin(); !i_player.Done(); i_player++) {
        //Debug(String("GameFinished player: ") + (*i_player)->ToString());
        //Debug(String("GameFinished player isAllin: ") + (*i_player)->IsAllin());
        //Debug(String("GameFinished player isFolded: ") + (*i_player)->IsFolded());
        if (!(*i_player)->IsAllin() && 
            !(*i_player)->IsFolded()
            ) {
            num_players++;
        }
    }
    return num_players;
}
