/*
 * gravity.h
 *
 *  Created on: Jun 25, 2015
 *      Author: nigul
 */

#ifndef PHYSICS_GRAVITY_H_
#define PHYSICS_GRAVITY_H_

#include "field.h"
#include "body.h"
#include "engine3d/geometry/vector.h"

namespace engine3d {

class Gravity: public Field {
public:
	Gravity(const Body& rSource);
	virtual ~Gravity();

	// @Override
	Vector GetForce(const Body& rBody);
private:
	const Body& mrSource;
};

} /* namespace engine3d */

#endif /* PHYSICS_GRAVITY_H_ */
