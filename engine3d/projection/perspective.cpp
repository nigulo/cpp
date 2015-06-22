/*
 * Perspective.cpp
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#include "perspective.h"
#include "engine3d/scenegraph/camera.h"
#include <math.h>

using namespace engine3d;

Perspective::Perspective(Program& rProgram,
			float viewAngle, float aspect, float near, float far) :
		Projection(rProgram, near, far),
	    mViewAngle(viewAngle),
	    mAspect(aspect) {
	Update();
}

Perspective::~Perspective() {
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

void Perspective::Update(int width, int height) {
	mAspect = (float) width * Projection::GetFOVRatio() / height;
	Update();
}

void Perspective::Update() {
	float f = 1 / tan(M_PI * mViewAngle / 180 / 2);
	float md = mZFar - mZNear;
	mMatrix.Set(0, 0, f / mAspect); mMatrix.Set(0, 1, 0); mMatrix.Set(0, 2, 0);                 mMatrix.Set(0, 3, 0);
	mMatrix.Set(1, 0, 0);          mMatrix.Set(1, 1, f); mMatrix.Set(1, 2, 0);                 mMatrix.Set(1, 3, 0);
	mMatrix.Set(2, 0, 0);          mMatrix.Set(2, 1, 0); mMatrix.Set(2, 2, (mZFar + mZNear) / md); mMatrix.Set(2, 3, -2 * mZFar * mZNear / md);
	mMatrix.Set(3, 0, 0);          mMatrix.Set(3, 1, 0); mMatrix.Set(3, 2, 1);                mMatrix.Set(3, 3, 0);
}
