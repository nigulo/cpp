/*
 * Field.h
 *
 *  Created on: Jun 25, 2015
 *      Author: nigul
 */

#ifndef PHYSICS_FIELD_H_
#define PHYSICS_FIELD_H_

#include <vector>
#include "engine3d/geometry/vector.h"
#include "body.h"

using namespace std;

namespace engine3d {

class Field {
public:
	Field();
	virtual ~Field();

	void AddBody(Body* pBody);
	/**
	 * General rule is following:
	 * if mBodies is empty the field applies to all
	 * bodies, otherwise only to the bodies in the list
	 */
	bool ApplicableTo(const Body& rBody);

	virtual Vector GetForce(const Body& rBody) = 0;
private:
	vector<Body*> mBodies;
};

} /* namespace engine3d */

#endif /* PHYSICS_FIELD_H_ */
