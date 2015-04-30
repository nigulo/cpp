#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "vector.h"

using namespace base;

namespace engine3d {

class Triangle
{
	public:
		Triangle(const Vector& v1, const Vector& v2, const Vector& v3);
		~Triangle();
		const Vector& GetVector(int no) const;
		/**
		 * @return the normal of the triangle. The normal points 
         * towards the movement of the screw which is rotated in 
         * the direction point 1 -> point 2 -> point 3
		 **/
		Vector GetNormal() const;
    private:
        Vector v1;
        Vector v2;
        Vector v3;
};
}
#endif // TRIANGLE_H
