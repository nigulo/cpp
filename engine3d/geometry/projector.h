#ifndef GEOMETRY_PROJECTOR_H
#define GEOMETRY_PROJECTOR_H

#include "plane.h"
#include <vector>

using namespace std;

namespace engine3d {
/*
 * Implements the parallel projection of a shape to a given plane
 */
class Projector
{
public:
	Projector();
	~Projector();
	/**
	 * Adds the new projection plane.
	 * point - the coordinates of an arbitrary point in plane
	 * normal - normal vector of the plane
	 */
	void AddPlane(const Vector& point, const Vector& normal);
	void AddPlane(const Plane& plane);
	/**
	 * Sets the projection direction. The direction vector must
	 * point towards the projection planes.
	 */
	void SetDirection(const Vector& direction);
	Vector Project(const Vector& v) const;
protected:
	vector<Plane*> planes;
	Vector* d;
};
}
#endif // GEOMETRY_PROJECTOR_H
