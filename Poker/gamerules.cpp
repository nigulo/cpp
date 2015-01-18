#include "gamerules.h" // class's header file

using namespace poker;

int GameRules::CountEquals(int value, Card* cards[], int length) {
	int count = 0;
	for (int i = 0; i < length; i++) {
		if (cards[i]->Value() == value) {
			count++;
		}
	}
	return count;
}

bool GameRules::HasPair(int value, Card* cards[], int length) {
    if (length < 2) {
        return false;
    }

	if (CountEquals(value, cards, length) >= 2) {
		return true;
	}
	return false;
}

long GameRules::HasPair(Card* cards[], int length) {
    if (length < 2) {
        return 0;
    }

	for (int i = 12; i >= 0; i--) {
		if (HasPair(i, cards, length)) {
			bool used[length];
			for (int j = 0; j < length; j++) {
				if (cards[j]->Value() == i) {
					used[j] = true;
				}
				else {
					used[j] = false;
				}
			}
		
			return (((long) i) << 12) | GetHandPoints(cards, used, length);
		}
	}
	return 0;
}

bool GameRules::HasThreeOfAKind(int value, Card* cards[], int length) {
    if (length < 3) {
        return false;
    }

	if (CountEquals(value, cards, length) >= 3) {
		return true;
	}
	return false;
}

long GameRules::HasThreeOfAKind(Card* cards[], int length) {
    if (length < 3) {
        return 0;
    }

	for (int i = 12; i >= 0; i--) {
		if (HasThreeOfAKind(i, cards, length)) {
			bool used[length];
			for (int j = 0; j < length; j++) {
				if (cards[j]->Value() == i) {
					used[j] = true;
				}
				else {
					used[j] = false;
				}
			}
		
			return (((long) i) << 8) | GetHandPoints(cards, used, length);
		}
	}
	return 0;
}


bool GameRules::HasFourOfAKind(int value, Card* cards[], int length) {
    if (length < 4) {
        return false;
    }

	if (CountEquals(value, cards, length) == 4) {
		return true;
	}
	return false;
}

long GameRules::HasFourOfAKind(Card* cards[], int length) {
    if (length < 4) {
        return 0;
    }

	for (int i = 12; i >= 0; i--) {
		if (HasFourOfAKind(i, cards, length)) {
			bool used[length];
			for (int j = 0; j < length; j++) {
				if (cards[j]->Value() == i) {
					used[j] = true;
				}
				else {
					used[j] = false;
				}
			}
		
			return (((long) i) << 4) | GetHandPoints(cards, used, length);
		}
	}
	return 0;
}


bool GameRules::HasTwoPair(int value1, int value2, Card* cards[], int length) {
    if (length < 4) {
        return false;
    }

	if (CountEquals(value1, cards, length) == 2 && CountEquals(value2, cards, length) == 2) {
		return true;
	}
	return false;
}

long GameRules::HasTwoPair(Card* cards[], int length) {
    if (length < 4) {
        return 0;
    }

	for (int i = 12; i >= 0; i--) {
		for (int j = i - 1; j >= 0; j--) {
			//if (j == i) {
			//	continue;
			//}
			if (HasTwoPair(i, j, cards, length)) {
				bool used[length];
				for (int k = 0; k < length; k++) {
					if (cards[k]->Value() == i || cards[k]->Value() == j) {
						used[k] = true;
					}
					else {
						used[k] = false;
					}
				}
			
				return (i > j ? (((long) i) << 8) + (j << 4) : (((long) j) << 8) + (((long) i) << 4)) | GetHandPoints(cards, used, length);
			}
		}
	}
	return 0;
}


bool GameRules::HasFullHouse(int value1, int value2, Card* cards[], int length) {
    if (length < 5) {
        return false;
    }

	if (CountEquals(value1, cards, length) == 3 && CountEquals(value2, cards, length) == 2) {
		return true;
	}
	return false;
}

long GameRules::HasFullHouse(Card* cards[], int length) {
    if (length < 5) {
        return 0;
    }

	for (int i = 12; i >= 0; i--) {
		for (int j = 12; j >= 0; j--) {
			if (j == i) {
				continue;
			}
			if (HasFullHouse(i, j, cards, length)) {
				return ((((long) i) << 4) + j);
			}
		}
	}
	return 0;
}


long GameRules::HasStraight(Card* cards[], int length) {
    if (length < 5) {
        return 0;
    }

	int order[length];
    SortCards(cards, length, order);

	int num = 1;
	int maxnum = 1;
	long value = 0;
	for (int i = 0; i < length - 1; i++) {
		if (cards[order[i]]->Value() == cards[order[i + 1]]->Value() + 1) {
  			num++;
  			if (num > maxnum) {
  				maxnum = num;
  				value = cards[order[i + 1]]->Value();
  				if (maxnum == 5) {
  					break;
  				}
  			}
  		}
  		else if (cards[order[i]]->Value() != cards[order[i + 1]]->Value()){
  			num = 1;
  		}
	}
	if (maxnum == 4 && cards[order[0]]->Value() == 12 && value == 0) {
		value = -1;
		maxnum++;
	}
	if (maxnum == 5) {
		return value + 2;
	}
	else {
    	return 0;
    }
}

long GameRules::HasFlush(Card* cards[], int length) {
    if (length < 5) {
        return 0;
    }

	int order[length];
    SortCards(cards, length, order);
	for (int i = 0; i < length; i++) {
        //cout << cards[order[i]]->ToString() << "\n";
    }
	for (int j = 0; j < 4; j++) {
		int num = 0;
		long points = 0;
		for (int i = 0; i < length; i++) {
            //cout << cards[order[i]]->ToString() << "\n";
			if (cards[order[i]]->Suit() == j) {
                //cout << cards[order[i]]->ToString() << "\n";
				num++;
				points += cards[order[i]]->Value() << (4 * (5 - num));
				if (num == 5) {
					return points;
				}
			}
		}
	}
	return 0;
}

long GameRules::HasStraightFlush(Card* cards[], int length) {
    if (length < 5) {
        return 0;
    }
    
	Card* flushCards[length];
	//for (int i = 0; i < length; i++) {
    //    flushCards[i] = 0;
    //}
	int flushLength = GetMaxNumSuited(cards, length, flushCards);
	long straightPoints = 0;
	if (flushLength >= 5) {
        straightPoints = HasStraight(flushCards, flushLength);
	}
	//for (int i = 0; i < length; i++) {
    //    if (flushCards[i]) {
    //    	delete flushCards[i];
    //    }
    //}
	return straightPoints;
}

long GameRules::GetHandPoints(Card* cards[], bool used[], int length) {
	long points = 0;
	int num_used = 0;
	//for (int i = 0; i < length; i++) {
	//	if (used[i]) {
	//		numUsed++;
	//	}
	//}
	bool first_time = true;
	int num_to_use = 5;
	if (length < num_to_use) {
        num_to_use = length;
    }
	while (num_used < num_to_use) {
		Card* highest = 0;
		int index = 0;
		for (int i = 0; i < length; i++) {
            if (!used[i]) {
    			if (highest == 0 || cards[i]->Value() > highest->Value()) {
    				highest = cards[i];
    				index = i;
    			}
            }
            else if (first_time) {
		        num_used++;
            }
		}
		if (num_used >= num_to_use) {
            // all cards already used
            break;
        }
		first_time = false;
		used[index] = true;
		num_used++;
		points += cards[index]->Value() << (4 * (num_to_use - num_used));
	}
	return points;
}

long GameRules::GetHandStrength(Card* cards[], int length) {
    //Dbg(String("num cards: ") + length);

	long strength = 0;
	long points = 0;
	
	if (points = HasPair(cards, length)) {
        //Dbg("has pair");
		strength = 1;
		long points1 = HasTwoPair(cards, length);
		if (points1) {
            //Dbg("has two pair");
			points = points1;
			strength = 2;
		}
		points1 = HasThreeOfAKind(cards, length);
		if (points1) {
            //Dbg("has three of a kind");
			points = points1;
			strength = 3;
			points1 = HasFullHouse(cards, length);
			if (points1) {
            //Dbg("has full house");
				points = points1;
				strength = 6;
			}
			points1 = HasFourOfAKind(cards, length);
			if (points1) {
                //Dbg("has four of a kind");
				points = points1;
				strength = 7;
			}
		}
	}
	
	if (strength < 4) {
		long points1 = HasStraight(cards, length);
		if (points1) {
            //Dbg("has straight");
			points = points1;
			strength = 4;
		}
	}
	
	long points1 = HasFlush(cards, length);
	if (points1) {
		long points2 = HasStraightFlush(cards, length);
		if (points2) {
            //Dbg("has straight flush");
			points = points2;
			strength = 8;
		}
		else if (strength < 5) {
            //Dbg("has flush");
			points = points1;
			strength = 5;
		}
	}
	
	if (strength == 0) {
        //Dbg("has high card");
		bool used[length];
		for (int i = 0; i < length; i++) {
			used[i] = false;
		}
		points = GetHandPoints(cards, used, length);
	}

	return (strength << 20) | points;
}

int GameRules::GetMaxNumSuited(Card* cards[], int length, Card* cardsOut[]) {
	for (int j = 0; j < 4; j++) {
		int num = 0;
		//int points = 0;
		//for (int i = 0; i < length; i++) {
        //    cardsOut[i] = 0;
        //}
		for (int i = 0; i < length; i++) {
			if (cards[i]->Suit() == j) {
				cardsOut[num++] = cards[i];
				//points += cards[i]->Value();
			}
		}
		if (num >= 5) {
    		return num;
        }
		//for (int i = 0; i < length; i++) {
        //    if (cardsOut[i]) {
        //        delete cardsOut[i];
        //    }
        //}
	}
	return 0;
}

void GameRules::SortCards(Card* cards[], int length, int indices[]) {
	bool used[length];
	for (int i = 0; i < length; i++) {
		used[i] = false;
	}
	for (int j = 0; j < length; j++) {
		Card* highest = 0;
		int index = 0;
		for (int i = 0; i < length; i++) {
			if (!used[i] && (highest == 0 || cards[i]->Value() > highest->Value())) {
				highest = cards[i];
				index = i;
			}
		}
		indices[j] = index;
		used[index] = true;
	}
}
