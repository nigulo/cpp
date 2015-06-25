#include "plane.h"
#include "triangle.h"

using namespace engine3d;

/** 
 * @point the coordinates of an arbitrary point in plane
 * @normal normal vector of the plane
 **/
Plane::Plane(const Vector& rPoint, const Vector& rNormal) :
    mPoint(rPoint),
    mNormal(Vector(rNormal).Normalize())
{
}

Plane::Plane(const Vector& rPoint1, const Vector& rPoint2, const Vector& rPoint3) :
    mPoint(rPoint1),
    mNormal(Triangle(rPoint1, rPoint2, rPoint3).GetNormal())
{
	//Object::Dbg("Creating plane from three points " + rPoint1.ToString() + ", " + rPoint2.ToString() + ", " + rPoint3.ToString());
}


Plane::~Plane()
{
}

const Vector& Plane::GetPoint() const 
{
	return mPoint;
}

const Vector& Plane::GetNormal() const 
{
	return mNormal;
}

int Plane::WhichSide(const Vector& rPoint) const
{
    double distance = (rPoint - mPoint).DotProduct(mNormal);
    if (distance > 0) {
        return PLACEMENT_FRONT;
    }
    else if (distance < 0) {
        return PLACEMENT_BACK;
    }
    else {
        return PLACEMENT_COINCIDE;
    }
}

double Plane::GetDistance(const Vector& rPoint) const
{
    return (mNormal * (rPoint - mPoint).DotProduct(mNormal)).Length();
}

Vector Plane::GetDistanceVector(const Vector& rPoint) const
{
    return mNormal * (rPoint - mPoint).DotProduct(mNormal);
}

string Plane::ToString() const {
	string ret_val = "Plane " + mPoint.ToString() + ", " + mNormal.ToString();
	return ret_val;
}
