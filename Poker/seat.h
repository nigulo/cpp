#ifndef SEAT_H
#define SEAT_H

#include "base/object.h" // inheriting class's header file
#include "base/string.h"
#include "player.h"

namespace poker {
    
    class Table;
    /**
     * No description
     */
    class Seat : public Object
    {
        
        private:
            int mNumber;
            Table& mrTable;
            Player* mpPlayer;
            
        public:
            
            Seat(Table& rTable, int number) :
                mrTable(rTable),
                mNumber(number),
                mpPlayer(0) {
            }
            
            String ToString() const;

            /**
             * Compares two seats.
             * @return true if this seat belongs to the
             * same table that the given seat and
             * if the seat locations match
             **/
            bool operator==(const Seat& rSeat) {
                return (mNumber == rSeat.mNumber && &mrTable == &rSeat.mrTable);
            }
            
            /**
             * Sits the given player on this seat
             * if the seat is empty
             **/
            void SitDown(Player& rPlayer) {
                if (mpPlayer) {
                    throw (String("Cannot sit down ") + rPlayer.ToString() + ", seat already taken");
                }
                mpPlayer = &rPlayer;
                Debug(String("Player ") + rPlayer.ToString() + " seated");
            }

            void StandUp(Player& rPlayer) {
                if (mpPlayer != &rPlayer) {
                    throw (String("Cannot stand up ") + rPlayer.ToString());
                }
                Debug(String("Player ") + rPlayer.ToString() + " stood up");
                mpPlayer = 0;
            }
            
            /**
             * @return the pointer to the player
             * who is sitting on this seat or 0
             * if the seat is not taken
             **/
            Player* GetPlayer() const {
                return mpPlayer;
            }
    
            int Number() const {
                return mNumber;
            }
            
            /**
             * @return the reference to the table
             * object this seat is belonging to
             **/
            Table& GetTable() const {
                return mrTable;
            }
    };
}
#endif // SEAT_H
