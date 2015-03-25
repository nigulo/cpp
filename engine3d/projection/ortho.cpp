/*
 * ortho.cpp
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#include "ortho.h"
#include "engine3d/scenegraph/camera.h"
#include <gl.h>

using namespace engine3d;

Ortho::Ortho(double left, double right, double bottom, double top, double near, double far) :
	Projection(near, far),
    mLeft(left),
    mRight(right),
    mBottom(bottom),
    mTop(top) {
}

Ortho::~Ortho() {
}

void Ortho::Init() const {
	Projection::Init();
    glOrtho(mLeft, mRight, mBottom, mTop, mZNear, mZFar);
}

const Plane Ortho::GetTopPlane(const Camera& r_camera) const {
	return Plane(r_camera.GetEye() + r_camera.GetUp() * mTop, -r_camera.GetUp());
}

const Plane Ortho::GetBottomPlane(const Camera& r_camera) const {
	return Plane(r_camera.GetEye() + (-r_camera.GetUp()) * mBottom, r_camera.GetUp());
}

const Plane Ortho::GetLeftPlane(const Camera& r_camera) const {
	Vector right = r_camera.GetDirection().CrossProduct(r_camera.GetUp());
	return Plane(r_camera.GetEye() + (-right) * mLeft, right);
}

const Plane Ortho::GetRightPlane(const Camera& r_camera) const {
	Vector left = r_camera.GetUp().CrossProduct(r_camera.GetDirection());
	return Plane(r_camera.GetEye() + (-left) * mRight, left);
}
