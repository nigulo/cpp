#ifndef CARD_H
#define CARD_H

#include <iostream> 
#include "base/object.h"

using namespace std;
using namespace base;

namespace poker {

    /**
     * No description
     */
    class Card : public Object
    {
        public:
    
            static const char msSuitNames[];
            static const char msSuitNamesLow[];
            static const char msValueNames[];
            static const char msValueNamesLow[];
            static const char msHandNames[10][20];
    
        private:
            int mSuit;
            int mValue;

        public:
                
            /**
             * Creates a new card using given suit and value
             **/
            Card(int suit, int value);

            /**
             * Creates a new card using given card string
             **/
            Card(char label[]);
            
            Card(const Card& rCard);
            Card& operator=(const Card& rCard);
            bool operator==(const Card& rCard);
            
            
            int Suit() const {
                return mSuit;
            }

            int Value() const {
                return mValue;
            }
            
            String ToString() const {
                char str[3] = {0, 0, 0};
            	str[0] = msSuitNames[mSuit];
            	str[1] = msValueNames[mValue];
                return String(str);
            }
            
            friend ostream& operator<<(ostream& rOut, const Card& rCard) {
                char str[3] = {0, 0, 0};
            	str[0] = msSuitNames[rCard.mSuit];
            	str[1] = msValueNames[rCard.mValue];
                return rOut << str;
            }
            
    };
}
#endif // CARD_H
