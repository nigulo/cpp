/*
 * Projection.cpp
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#include "projection.h"
#include <GL/gl.h>

using namespace engine3d;

Projection::Projection(double near, double far) :
	mZNear(near),
	mZFar(far),
	mMatrix(4) {

}

Projection::~Projection() {
}

void Projection::Project() const {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	glMultMatrixd(mMatrix.GetElements());
}
