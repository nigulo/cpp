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

unique_ptr<Vector> BoundingSphere::Intersects(const Line& rLine) const
{
    if (rLine.GetDistance(mTransformedPosition) <= mRadius) {
        if (!rLine.IsSegment()) {
            Debug("not segment");
            // TODO correct this
            return unique_ptr<Vector>(new Vector(mPosition));
        }
        else {
            if ((rLine.mPoint1 - mTransformedPosition).Length() <= mRadius ||
                (rLine.mPoint2 - mTransformedPosition).Length() <= mRadius
            ) {
                Debug("segment");
                // TODO correct this
                return unique_ptr<Vector>(new Vector(rLine.mPoint1));
            }
            else {
                return unique_ptr<Vector>();
            }
        }
    }
    else {
        return unique_ptr<Vector>();
    }
}

unique_ptr<Vector> BoundingSphere::Collides(const BoundingVolume& rOtherBound) const
{
    Debug("Sphere.Collide 1");
    const Vector& old_pos = mTransformedPosition;//mTransformation.Transform(mPosition);
    Vector new_pos = GetTransformation().Transform(mPosition);
    Debug(string("Sphere old pos: ") + old_pos.ToString());
    Debug(string("Sphere new pos: ") + new_pos.ToString());
    Vector v = (new_pos - old_pos).Normalize() * mRadius;
    Segment movement_line(old_pos - v, new_pos + v);
    Debug("Sphere.Collide 2");
    unique_ptr<Vector> intersection_point = rOtherBound.Intersects(movement_line);
    if (intersection_point.get()) {
        // movement line of the center of sphere intersects
        return intersection_point;
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
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular, new_pos + perpendicular));
    if (intersection_point.get()) {
        // 1st point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 4");
    perpendicular = -perpendicular;
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular, new_pos + perpendicular));
    if (intersection_point.get()) {
        // 2nd point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 5");
    Vector perpendicular2 = movement_line.mDirection.CrossProduct(perpendicular);
    perpendicular2.Normalize();
    perpendicular2 *= mRadius;
    Debug("Sphere.Collide 6");
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular2, new_pos + perpendicular2));
    if (intersection_point.get()) {
        // 3rd point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 7");
    perpendicular2 = -perpendicular2;
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular2, new_pos + perpendicular2));
    if (intersection_point.get()) {
        // 4th point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 8");
    Vector perpendicular3 = perpendicular + perpendicular2;
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3));
    if (intersection_point.get()) {
        // 5th point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 9");
    perpendicular3 = perpendicular - perpendicular2;
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3));
    if (intersection_point.get()) {
        // 6th point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 10");
    perpendicular3 = -perpendicular + perpendicular2;
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3));
    if (intersection_point.get()) {
        // 7th point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 11");
    perpendicular3 = -(perpendicular + perpendicular2);
    perpendicular3.Normalize();
    perpendicular3 *= mRadius;
    intersection_point = rOtherBound.Intersects(Segment(old_pos + perpendicular3, new_pos + perpendicular3));
    if (intersection_point.get()) {
        // 8th point on sphere surface intersects
        return intersection_point;
    }
    Debug("Sphere.Collide 12");
    return unique_ptr<Vector>();
}

void BoundingSphere::Transform()
{
    Spatial::Transform();
    mTransformedPosition = GetOldTransformation().Transform(mPosition);
}
