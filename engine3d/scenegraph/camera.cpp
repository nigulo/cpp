#include "camera.h"
#include <math.h>
#include <cassert>
#include <GL/glew.h>

using namespace engine3d;

Camera::Camera(Program& rProgram, Projection* pProjection) : Node("Camera"),
		mpProjection(pProjection),
		mrUniform(rProgram.GetUniform("mv_matrix")),
		mEye(0, 0, 0),
		mCenter(0, 0, 1),
		mUp(0, 1, 0), // must be normalized
		mMatrix(4) {
	assert(pProjection);
	assert(&rProgram == &pProjection->GetProgram());
	Update();
}

Camera::~Camera()
{
}

void Camera::Render()
{
	Debug(string("Camera::Render eye ") + mEye.ToString());
	Debug(string("Camera::Render center ") + mCenter.ToString());
	Debug(string("Camera::Render up ") + mUp.ToString());
	mpProjection->Project();
	string log = "glModel view matrix: ";
	for (int i = 0; i < mMatrix.GetNumRows() * mMatrix.GetNumColumns(); i++) {
		log += " " + to_string(mMatrix.GetElements()[i]);
	}
	Debug(log);
    glUniformMatrix4fv(
        mrUniform.GetId(),
        1, GL_FALSE,
        mMatrix.GetElements()
    );
    base::Object::Dbg(string("glUniformMatrix4fv(") + to_string(mrUniform.GetId()) + ", 1, GL_FALSE, _)");
}

/**
 * The normal of the returned plane is pointing inside
 * the viewing frustrum
 * Not precise!
 **/
const Plane Camera::GetFarPlane() const {
	Vector plane_normal = GetDirection();
	//Vector planePoint = eye + (planeNormal * zFar);
	// NB! This is not correct. just added to avoid 
    // distantprojections rounded behind farplane
	Vector plane_point = mEye + (plane_normal * mpProjection->GetZFar() * 0.99);
	return Plane(plane_point, plane_normal * (-1));
}

/**
 * The normal of the returned plane is pointing inside
 * the viewing frustrum
 * WARNING: NOT TESTED!!!
 **/
const Plane Camera::GetNearPlane() const {
	Vector plane_normal = GetDirection();
	Vector plane_point = mEye + (plane_normal * mpProjection->GetZNear());
	return Plane(plane_point, plane_normal);
}

/**
 * The normal of the returned plane is pointing inside
 * the viewing frustrum
 * WARNING: NOT TESTED!!!
 **/
const Plane Camera::GetTopPlane() const {
	return mpProjection->GetTopPlane(*this);
}

/**
 * The normal of the returned plane is pointing inside
 * the viewing frustrum
 * WARNING: NOT TESTED!!!
 **/
const Plane Camera::GetBottomPlane() const {
	return mpProjection->GetBottomPlane(*this);
}

/**
 * The normal of the returned plane is pointing inside
 * the viewing frustrum
 * WARNING: NOT TESTED!!!
 **/
const Plane Camera::GetLeftPlane() const {
	return mpProjection->GetLeftPlane(*this);
}

/**
 * The normal of the returned plane is pointing inside
 * the viewing frustrum
 * WARNING: NOT TESTED!!!
 **/
const Plane Camera::GetRightPlane() const {
	return mpProjection->GetRightPlane(*this);
}

/**
 * Sets the camera's world transformation. Transformation
 * defines the location and direction of the camera.
 * @param rTransformation new transformation of the camera. 
 **/
//void Camera::SetTransformation(const Transformation& rTransformation) {
//    // Re-calculate eye, center and up vectors and 
//    // set changed to true
//
//    if (!CheckCollisions(rTransformation)) {
//        Vector mEye = rTransformation.Transform(Vector(0, 0, 0));
//        Vector mCenter = rTransformation.Transform(Vector(0, 0, 1));
//        Vector mUp = (rTransformation.Rotate(Vector(0, 1, 0))).Normalize();
//        mTransformation = rTransformation;
//        mWorldTransformation = rTransformation;
//    }
//}

/**
 * Checks if the given volume is out of the camera's view.
 * @rVolume bounding volume to check for culling
 * @return ture if the given volume can be culled, false otherwise
 **/
bool Camera::Cull(const BoundingVolume& rVolume)
{
    return false; // Culling may not work correctly with ortographic projection
    if (rVolume.WhichSide(GetLeftPlane()) == Plane::PLACEMENT_BACK) {
        Debug("Cull 2");
        return true;
    }
    else if (rVolume.WhichSide(GetRightPlane()) == Plane::PLACEMENT_BACK) {
        Debug("Cull 3");
        return true;
    }
    else if (rVolume.WhichSide(GetTopPlane()) == Plane::PLACEMENT_BACK) {
        Debug("Cull 4");
        return true;
    }
    else if (rVolume.WhichSide(GetBottomPlane()) == Plane::PLACEMENT_BACK) {
        Debug("Cull 5");
        return true;
    }
    else if (rVolume.WhichSide(GetNearPlane()) == Plane::PLACEMENT_BACK) {
        Debug("Cull 6");
        return true;
    }
    else if (rVolume.WhichSide(GetFarPlane()) == Plane::PLACEMENT_BACK) {
        Debug("Cull 7");
        return true;
    }
    return false;
}

void Camera::Transform()
{
	Debug("Camera.Transform");
    Spatial::Transform();
    // Re-calculate eye, center and up vectors
    mEye = mTransformation.Transform(Vector(0, 0, 0));
    mCenter = mTransformation.Transform(Vector(0, 0, 1));
    mUp = (mTransformation.Rotate(Vector(0, 1, 0))).Normalize();
    Update();
}

void Camera::Update() {
	Vector f(mCenter - mEye);
	f.Normalize();
	Vector s = f.CrossProduct(mUp);
	Vector sNorm = s;
	sNorm.Normalize();
	Vector u = sNorm.CrossProduct(f);
	//mMatrix.Set(0, 0, s[0]);  mMatrix.Set(0, 1, s[1]);  mMatrix.Set(0, 2, s[2]);  mMatrix.Set(0, 3, -s.DotProduct(mEye));
	//mMatrix.Set(1, 0, u[0]);  mMatrix.Set(1, 1, u[1]);  mMatrix.Set(1, 2, u[2]);  mMatrix.Set(1, 3, -u.DotProduct(mEye));
	//mMatrix.Set(2, 0, -f[0]); mMatrix.Set(2, 1, -f[1]); mMatrix.Set(2, 2, -f[2]); mMatrix.Set(2, 3, f.DotProduct(mEye));
	//mMatrix.Set(3, 0, 0);     mMatrix.Set(3, 1, 0);     mMatrix.Set(3, 2, 0);     mMatrix.Set(3, 3, 1);

	mMatrix.Set(0, 0, 1);  mMatrix.Set(0, 1, 0);  mMatrix.Set(0, 2, 0);  mMatrix.Set(0, 3, 0);
	mMatrix.Set(1, 0, 0);  mMatrix.Set(1, 1, 1);  mMatrix.Set(1, 2, 0);  mMatrix.Set(1, 3, 0);
	mMatrix.Set(2, 0, 0);  mMatrix.Set(2, 1, 0); mMatrix.Set(2, 2, 1); mMatrix.Set(2, 3, 0);
	mMatrix.Set(3, 0, 0);  mMatrix.Set(3, 1, 0);     mMatrix.Set(3, 2, 0);     mMatrix.Set(3, 3, 1);
    mChanged = true;
}
