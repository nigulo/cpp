/*
 * collisionlistener.h
 *
 *  Created on: Jun 26, 2015
 *      Author: nigul
 */

#ifndef SCENEGRAPH_COLLISIONLISTENER_H_
#define SCENEGRAPH_COLLISIONLISTENER_H_

#include "engine3d/geometry/vector.h"
#include <memory>

namespace engine3d {

class Node;
class CollisionListener {
public:
	CollisionListener();
	virtual ~CollisionListener();

	virtual void Collision(const Node& rNode1, const Node& rNode2, const unique_ptr<Vector>& rPoint) = 0;
};

} /* namespace engine3d */

#endif /* SCENEGRAPH_COLLISIONLISTENER_H_ */
