// Class automatically generated by Dev-C++ New Class wizard

#include <gl.h>
#include <glu.h>

#include "camera.h" // class's header file
#include <math.h>

using namespace engine3d;
// class constructor
Camera::Camera(Projection* p_projection) : Node("Camera"),
    mpProjection(p_projection),
    mEye(0, 0, 0),
    mCenter(0, 0, 1),
    mUp(0, 1, 0) {
    mChanged = true;
	assert(mpProjection);
}

Camera::Camera(const Camera& rCam) :
    mpProjection(rCam.mpProjection),
    mEye(rCam.mEye),
    mCenter(rCam.mCenter),
    mUp(rCam.mUp),
    Node(rCam) {
    mChanged = true;
}

// class destructor
Camera::~Camera()
{
}

void engine3d::Camera::Init() {
	mpProjection->Init();
    glMatrixMode(GL_MODELVIEW);
    mChanged = true;
	Node::Init();
}

void Camera::Look()
{
	//if (mPerspective) {
	    //Vector eye = mTransformation.Transform(mEye);
	    //Vector center = mTransformation.Transform(mCenter);
	    //Vector up = mTransformation.Transform(mUp);
	    Debug(String("Camera::Look eye ") + mEye.ToString());
	    Debug(String("Camera::Look center ") + mCenter.ToString());
	    Debug(String("Camera::Look up ") + mUp.ToString());
	    gluLookAt(mEye[0], mEye[1], mEye[2],
	              mCenter[0], mCenter[1], mCenter[2],
	              mUp[0], mUp[1], mUp[2]);
	//}
    mChanged = false;
}

/**
 * The normal of the returned plane is pointing inside
 * the viewing frustrum
 * Not precise!
 **/
const Plane Camera::GetFarPlane() const {
    //assert(center);
    //assert(eye);
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
    //assert(center);
    //assert(eye);
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
    Spatial::Transform();
    // Re-calculate eye, center and up vectors
    mEye = mTransformation.Transform(Vector(0, 0, 0));
    mCenter = mTransformation.Transform(Vector(0, 0, 1));
    mUp = (mTransformation.Rotate(Vector(0, 1, 0))).Normalize();
}
