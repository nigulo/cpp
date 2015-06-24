#include "triangle.h"
#include <cassert>

using namespace engine3d;

Triangle::Triangle(const Vector& vector1, const Vector& vector2, const Vector& vector3) : 
    v1(vector1), v2(vector2), v3(vector3)
{
}

Triangle::~Triangle()
{
}

const Vector& Triangle::GetVector(int no) const
{
    assert(no >= 0 && no <= 2);
	switch (no) {
        case 0:
            return v1;
        case 1:
            return v2;
        case 2:
            return v3;
    }
}

Vector Triangle::GetNormal() const
{
    Vector v12 = v2 - v1;
    Vector v13 = v3 - v1;
    return (v12.CrossProduct(v13)).Normalize();
}

string Triangle::ToString() const {
	string ret_val = "Triangle " + v1.ToString() + ", " + v2.ToString() + ", " + v3.ToString();
	return ret_val;
}
