#include "projector.h"
#include <cassert>

using namespace engine3d;

Projector::Projector() :
    d(0),
    planes(1)
{
}

Projector::~Projector()
{
	for (int i = 0; i < planes.size(); i++) {
        delete planes[i];
    }
    if (d) {
        delete d;
    }
    planes.clear();
}

/*
 * Adds the new projection plane.
 * point - the coordinates of an arbitrary point in plane
 * normal - normal vector of the plane
 */
void Projector::AddPlane(const Vector& point, const Vector& normal)
{
	planes.push_back(new Plane(point, normal));
}

void Projector::AddPlane(const Plane& plane)
{
	planes.push_back(new Plane(plane.GetPoint(), plane.GetNormal()));
}

/*
 * Sets the projection direction. The direction vector must 
 * point towards the projection planes.
 */
void Projector::SetDirection(const Vector& direction)
{
    if (d) {
        delete d;
    }
	d = new Vector(direction);
	d->Normalize();
}

Vector Projector::Project(const Vector& v) const
{
    assert(planes.size() > 0);
    Vector n = planes[0]->GetNormal();
    if (n.DotProduct(*d) < 0) {
    	n = n * -1.0;
    }
    Vector p = planes[0]->GetPoint();
    double nxr = n.DotProduct(p);
    double nxr0 = n.DotProduct(v);
    double nxd = n.DotProduct(*d);
    Vector d1 = n * (nxr - nxr0);
    return v + ((*d) * (d1.Length() / nxd));
}
