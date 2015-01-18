#include "card.h" // class's header file

using namespace poker;

const char Card::msSuitNames[] = "CDHS";
const char Card::msSuitNamesLow[] = "cdhs";
const char Card::msValueNames[] = "23456789TJQKA";
const char Card::msValueNamesLow[] = "23456789tjqka";
const char Card::msHandNames[10][20] = {
	"        No hand",
	"           Pair",
	"       Two pair",
	"Three of a kind",
	"       Straight",
	"          Flush",
	"     Full house",
	" Four of a kind",
	" Straight flush"
};

Card::Card(int suit, int value) :
    mSuit(suit),
    mValue(value)
{
}

Card::Card(char label[])
{
	for (int i = 0; i < 4; i++) {
    	if (label[0] == msSuitNames[i] || label[0] == msSuitNamesLow[i]) {
    		mSuit = i;
    		break;
		}
	}
	
	for (int i = 0; i < 13; i++) {
    	if (label[1] == msValueNames[i] || label[1] == msValueNamesLow[i]) {
    		mValue = i;
    		break;
		}
	}
	
	//std::cout << "suit = " << suit << ", value = " << value << "\n";
	
}

Card::Card(const Card& rCard)
{
    mSuit = rCard.mSuit;
    mValue = rCard.mValue;
}

Card& Card::operator=(const Card& rCard)
{
    mSuit = rCard.mSuit;
    mValue = rCard.mValue;
    return *this;
}

bool Card::operator==(const Card& rCard)
{
    return (mSuit == rCard.mSuit && mValue == rCard.mValue);
}

//char* Card::ToString() {
//    char str[3];
//	str[0] = 0;
//	str[1] = 0;
//	str[2] = 0;
//	str[0] = SUIT_NAMES[mSuit];
//	str[1] = VALUE_NAMES[mValue];
//	return str;
//}
