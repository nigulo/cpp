/*
 * Perspective.cpp
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#include "perspective.h"
#include "engine3d/scenegraph/camera.h"
#include <glu.h>
#include <math.h>

using namespace engine3d;

Perspective::Perspective(double viewAngle, double aspect, double near, double far) :
		Projection(near, far),
	    mViewAngle(viewAngle),
	    mAspect(aspect) {
}

Perspective::~Perspective() {
}

void Perspective::Init() const {
	Projection::Init();
	gluPerspective(mViewAngle, mAspect, mZNear, mZFar);
}

const Plane Perspective::GetTopPlane(const Camera& r_camera) const {
	Vector direction = r_camera.GetDirection();
	Vector down = (-r_camera.GetUp()) * tan((M_PI * mViewAngle / 360));
	Vector plane_normal = (direction + down) * 0.5;
	Vector plane_point = r_camera.GetEye();
	return Plane(plane_point, plane_normal);
}

const Plane Perspective::GetBottomPlane(
		const Camera& r_camera) const {
	Vector direction = r_camera.GetDirection();
	Vector plane_normal = (direction + r_camera.GetUp() * tan((M_PI * mViewAngle / 360))) * 0.5;
	Vector plane_point = r_camera.GetEye();
	return Plane(plane_point, plane_normal);
}

const Plane Perspective::GetLeftPlane(const Camera& r_camera) const {
	Vector direction = r_camera.GetDirection();
	Vector right = direction.CrossProduct(r_camera.GetUp());
	Vector plane_normal = (direction + right * tan((M_PI * mViewAngle * mAspect / 360))) * 0.5;
	Vector plane_point = r_camera.GetEye();
	return Plane(plane_point, plane_normal);
}

const Plane Perspective::GetRightPlane(const Camera& r_camera) const {
	Vector direction = r_camera.GetDirection();
	Vector left = r_camera.GetUp().CrossProduct(direction);
	Vector plane_normal = (direction + left * tan((M_PI * mViewAngle * mAspect / 360))) * 0.5;
	Vector plane_point = r_camera.GetEye();
	return Plane(plane_point, plane_normal);
}
