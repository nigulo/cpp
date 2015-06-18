#ifndef BOUNDINGSPHERE_H
#define BOUNDINGSPHERE_H

#include "boundingvolume.h"
#include "engine3d/geometry/vector.h"

using namespace base;

namespace engine3d {

class BoundingSphere : public BoundingVolume
{
	public:

		BoundingSphere(const Vector& rPos, float radius);
		~BoundingSphere();

		BoundingSphere* Clone() const;
		
		int WhichSide(const Plane& rPlane) const;

        bool Intersects(const Line& rLine) const;
        
   		bool Collides(const BoundingVolume& rOtherBound) const;

        void Transform();
	private:
        Vector mPosition;
        float mRadius;
        Vector mTransformedPosition;
};
}
#endif // BOUNDINGSPHERE_H
