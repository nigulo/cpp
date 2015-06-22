#include "boundingsphere.h"
#include "engine3d/geometry/segment.h"
#include <cassert>

using namespace engine3d;

BoundingSphere::BoundingSphere(const Vector& rPos, float radius) :
    mPosition(rPos),
    mTransformedPosition(rPos)
{
    Debug(string("sphere pos:") + mPosition.ToString());
    assert(radius >= 0);
    mRadius = radius;
}

BoundingSphere* BoundingSphere::Clone() const 
{
    BoundingSphere* p_bound = new BoundingSphere(mPosition, mRadius);
    return p_bound;
}

BoundingSphere::~BoundingSphere()
{
}

int BoundingSphere::WhichSide(const Plane& rPlane) const
{
    Vector distance = rPlane.GetDistanceVector(mTransformedPosition);
    if (distance.Length() < mRadius) {
        return Plane::PLACEMENT_COINCIDE;
    }
    else {
        return rPlane.WhichSide(mTransformedPosition);
    }
}

bool BoundingSphere::Intersects(const Line& rLine) const
{
    if (rLine.GetDistance(mTransformedPosition) <= mRadius) {
        if (!rLine.IsSegment()) {
            Debug("not segment");
            return true;
        }
        else {
            if ((rLine.mPoint1 - mTransformedPosition).Length() <= mRadius ||
                (rLine.mPoint2 - mTransformedPosition).Length() <= mRadius
            ) {
                Debug("segment");
                return true;
            }
            else {
                return false;
            }
        }
    }
    else {
        return false;
    }
}

bool BoundingSphere::Collides(const BoundingVolume& rOtherBound) const
{
    Debug("Sphere.Collide 1");
    const Vector& old_pos = mTransformedPosition;//mTransformation.Transform(mPosition);
    Vector new_pos = GetTransformation().Transform(mPosition);
    Debug(string("Sphere old pos: ") + old_pos.ToString());
    Debug(string("Sphere new pos: ") + new_pos.ToString());
    Vector v = (new_pos - old_pos).Normalize() * mRadius;
    Segment movement_line(old_pos - v, new_pos + v);
    Debug("Sphere.Collide 2");
    if (rOtherBound.Intersects(movement_line)) {
        // movement line of the center of sphere intersects
        return true;
    }
    // create arbitrary perpendicular to center movement line,
    // create some movement lines for sphere surface and check if
    // they intersect with given bound
    Vector perpendicular(0, 0, 0);
    if (movement_line.mDirection[0] != 0) {
        perpendicular = Vector(-(movement_line.mDirection[1] + movement_line.mDirection[2]) / movement_line.mDirection[0], 1, 1);
    }
    else if (movement_line.mDirection[1] != 0) {
        perpendicular = Vector(1, -(movement_line.mDirection[0] + movement_line.mDirection[2]) / movement_line.mDirection[1], 1);
    }
    else {
        perpendicular = Vector(1, 1, -(movement_line.mDirection[0] + movement_line.mDirection[1]) / movement_line.mDirection[2]);
    }
    perpendicular.Normalize();
    perpendicular *= mRadius;
    Debug("Sphere.Collide 3");
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular, new_pos + perpendicular))) {
        // 1st point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 4");
    perpendicular = -perpendicular;
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular, new_pos + perpendicular))) {
        // 2nd point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 5");
    Vector perpendicular2 = movement_line.mDirection.CrossProduct(perpendicular);
    perpendicular2.Normalize();
    perpendicular2 *= mRadius;
    Debug("Sphere.Collide 6");
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular2, new_pos + perpendicular2))) {
        // 3rd point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 7");
    perpendicular2 = -perpendicular2;
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular2, new_pos + perpendicular2))) {
        // 4th point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 8");
    Vector perpendicular3 = perpendicular + perpendicular2;
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3))) {
        // 5th point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 9");
    perpendicular3 = perpendicular - perpendicular2;
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3))) {
        // 6th point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 10");
    perpendicular3 = -perpendicular + perpendicular2;
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3))) {
        // 7th point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 11");
    perpendicular3 = -(perpendicular + perpendicular2);
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    if (rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3))) {
        // 8th point on sphere surface intersects
        return true;
    }
    Debug("Sphere.Collide 12");
    return false;
}

void BoundingSphere::Transform()
{
    Spatial::Transform();
    mTransformedPosition = GetOldTransformation().Transform(mPosition);
}
