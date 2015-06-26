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
	if (&rBody != &mrSource) { // no self gravity
		if (ApplicableTo(rBody)) {
			Vector diff = mrSource.GetPosition() - rBody.GetPosition();
			Object::Dbg("Gravity::GetForce " + diff.ToString());
			float len2 = diff.Length2();
			if (len2 > 0) { // no gravity at origin
				return diff * rBody.GetMass() * mrSource.GetMass() / (len2 * sqrt(len2));
			}
		}
	}
	return Vector();
}
