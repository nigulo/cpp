/*
 * Field.cpp
 *
 *  Created on: Jun 25, 2015
 *      Author: nigul
 */

#include "field.h"

using namespace engine3d;

Field::Field() {

}

Field::~Field() {
}

void Field::AddBody(Body* pBody) {
	mBodies.push_back(pBody);
}

bool Field::ApplicableTo(const Body& rBody) {
	if (mBodies.empty()) {
		return true;
	}
	for (auto&& p_body : mBodies) {
		if (p_body == &rBody) {
			return true;
		}
	}
	return false;
}
