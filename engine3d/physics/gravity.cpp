/*
 * gravity.cpp
 *
 *  Created on: Jun 25, 2015
 *      Author: nigul
 */

#include "gravity.h"
#include <math.h>

using namespace engine3d;

Gravity::Gravity(const Body& rSource) :
		mrSource(rSource) {
}

Gravity::~Gravity() {
}

Vector Gravity::GetForce(const Body& rBody) {
	if (ApplicableTo(rBody)) {
		Vector diff = mrSource.GetPosition() - rBody.GetPosition();
		float len2 = diff.Length2();
		return diff * rBody.GetMass() * mrSource.GetMass() / (len2 * sqrt(len2));
	}
	return Vector(0, 0, 0);
}
