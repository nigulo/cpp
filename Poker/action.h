#ifndef ACTION_H
#define ACTION_H

#include "base/object.h" // inheriting class's header file
#include "enum.h"

using namespace base;

namespace poker {
    class Player;
    
    /**
     * No description
     */
    class Action : public Object
    {
        public:
            class Type : public Enum<String>
            {
                public:
                    static const Type msSmallBlind;
                    static const Type msBigBlind;
                    static const Type msCheck;
                    static const Type msBet;
                    static const Type msCall;
                    static const Type msRaise;
                    static const Type msFold;
                    static const Type msAllin;
                private:
                    Type(const String& rType) : Enum<String>(rType){}
            };
            
        private:
            Player& mrPlayer;
            const Type& mrType;

            /**
             * The sum of the bet.
             * For instance if the bet on table is 15,
             * player's previous bet is 10 and player
             * raises by 5 then the sum is 10
             **/
            const int mSum;

            /**
             * The current bet of player before this 
             * bet is made.
             **/
            const int mCurrentPlayerBet;

            /**
             * The current bet on table before this 
             * bet is made.
             **/
            const int mCurrentBet;

            /**
             * The previous action of the same player
             **/
            Action* mpPreviousPlayerAction;
                    
            /**
             * The previous action on table
             **/
            Action* mpPreviousAction;
        public:
            Action(Player& rPlayer, const Type& rType, int sum);
            virtual ~Action();
            
            int Sum() const {
                return mSum;
            }
            
            int CurrentPlayerBet() const {
                return mCurrentPlayerBet;
            }

            int CurrentBet() const {
                return mCurrentBet;
            }

            Player& GetPlayer() const {
                return mrPlayer;
            }
            
            const Type& GetType() const {
                return mrType;
            }
            
            Action* GetPreviousAction() const {
                return mpPreviousAction;
            }

            Action* GetPreviousPlayerAction() const {
                return mpPreviousPlayerAction;
            }
            
            String ToString() const;
            
    };
}
#endif // ACTION_H
