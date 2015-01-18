#ifndef HOLDEMRULES_H
#define HOLDEMRULES_H

#include "gamerules.h" // inheriting class's header file

namespace poker {
    /**
     * No description
     */
    class HoldemRules : public GameRules
    {
    	public:
            HoldemRules(Table& rTable) : GameRules(rTable) {}

            void DealPocketCards();
            void DealFlop();
            void DealTurn();
            void DealRiver();
            
            void DetermineWinners();
    };
}
#endif // HOLDEMRULES_H
