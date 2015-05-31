/*
 * Projection.cpp
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#include "projection.h"
#include <GL/glew.h>

using namespace engine3d;

Projection::Projection(Program& rProgram, float near, float far) :
	mrProgram(rProgram),
	mrUniform(rProgram.GetUniform("p_matrix")),
	mZNear(near),
	mZFar(far),
	mMatrix(4) {

}

Projection::~Projection() {
}

void Projection::Project() const {
    glUniformMatrix4fv(
        mrUniform.GetId(),
        1, GL_FALSE,
        mMatrix.GetElements()
    );
}
