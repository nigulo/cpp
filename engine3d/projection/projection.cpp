/*
 * Projection.cpp
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#include "projection.h"
#include <gl.h>

using namespace engine3d;

Projection::Projection(double near, double far) :
	mZNear(near),
	mZFar(far) {

}

Projection::~Projection() {
}

void Projection::Init() const {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}
