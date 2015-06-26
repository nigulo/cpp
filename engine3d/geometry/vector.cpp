#include "vector.h"
#include <math.h>
#include <cassert>

using namespace engine3d;

Vector::Vector(int dim, bool zeroCoords)
{
    this->mDim = dim;
    this->mpCoords = new float[dim];
    if (zeroCoords) {
        for (int i = 0; i < dim; i++) {
            mpCoords[i] = 0;
        }
    }
}

Vector::Vector(int dim, float coords[])
{
    this->mDim = dim;
    this->mpCoords = new float[dim];
    for (int i = 0; i < dim; i++) {
    	this->mpCoords[i] = coords[i];
    }
}

Vector::Vector(float x, float y, float z, float w)
{
    mDim = 4;
    mpCoords = new float[mDim];
    mpCoords[0] = x;
    mpCoords[1] = y;
    mpCoords[2] = z;
    mpCoords[3] = w;
}

Vector::Vector(float x, float y, float z)
{
    mDim = 3;
    mpCoords = new float[mDim];
    mpCoords[0] = x;
    mpCoords[1] = y;
    mpCoords[2] = z;
}

Vector::Vector(float x, float y)
{
    mDim = 2;
    mpCoords = new float[mDim];
    mpCoords[0] = x;
    mpCoords[1] = y;
}

Vector::Vector(float x)
{
    mDim = 1;
    mpCoords = new float[mDim];
    mpCoords[0] = x;
}

Vector::Vector(const Vector& v)
{
    mDim = v.mDim;
    mpCoords = new float[mDim];
    for (int i = 0; i < mDim; i++) {
    	mpCoords[i] = v.mpCoords[i];
    }
}

Vector& Vector::operator=(const Vector& v) {
    if (this != &v) {
        if (mpCoords) {
            delete [](mpCoords);
        }
        mDim = v.mDim;
        mpCoords = new float[mDim];
        for (int i = 0; i < mDim; i++) {
            mpCoords[i] = v.mpCoords[i];
        }
    }
    return *this;
}

Vector Vector::operator-() const {
    Vector v(mDim, false);//(*this);
    for (int i = 0; i < mDim; i++) {
        v.mpCoords[i] = -mpCoords[i];
    }
    return v;
}


float Vector::operator[](int i) const {
    assert(i >= 0 && i < mDim);
    return mpCoords[i];
}

Vector Vector::operator+(const Vector& v) const {
    assert(mDim == v.mDim);
    Vector result(mDim, false);
    for (int i = 0; i < mDim; i++) {
        result.mpCoords[i] = mpCoords[i] + v.mpCoords[i];
    }
    return result;
}

Vector Vector::operator-(const Vector& v) const {
    assert(mDim == v.mDim);
    Vector result(mDim, false);
    for (int i = 0; i < mDim; i++) {
        result.mpCoords[i] = mpCoords[i] - v.mpCoords[i];
    }
    return result;
}

Vector Vector::operator*(float c) const {
    Vector result(mDim, false);
    for (int i = 0; i < mDim; i++) {
        result.mpCoords[i] = mpCoords[i] * c;
    }
    return result;
}

Vector Vector::operator/(float c) const {
    Vector result(mDim, false);
    for (int i = 0; i < mDim; i++) {
        result.mpCoords[i] = mpCoords[i] / c;
    }
    return result;
}

Vector& Vector::operator+=(const Vector& v) {
    assert(mDim == v.mDim);
    for (int i = 0; i < mDim; i++) {
        mpCoords[i] += v.mpCoords[i];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& v) {
    assert(mDim == v.mDim);
    for (int i = 0; i < mDim; i++) {
        mpCoords[i] -= v.mpCoords[i];
    }
    return *this;
}

Vector& Vector::operator*=(float c) {
    for (int i = 0; i < mDim; i++) {
        mpCoords[i] *= c;
    }
    return *this;
}

Vector& Vector::operator/=(float c) {
    for (int i = 0; i < mDim; i++) {
        mpCoords[i] /= c;
    }
    return *this;
}

// class destructor
Vector::~Vector()
{
    if (mpCoords) {
	   delete []mpCoords;
    }
}

// returns the coordinate of this vector at the specified index
//float Vector::GetCoord(int index) const
//{
//    assert(dim > index);
//	return coords[index];
//}

void Vector::SetCoord(int index, float d)
{
	mpCoords[index] = d;
}

//float Vector::GetX() const
//{
//    assert(dim > 0);
//	return coords[0];
//}
//
//float Vector::GetY() const
//{
//    assert(dim > 1);
//	return coords[1];
//}
//
//float Vector::GetZ() const
//{
//    assert(dim > 2);
//	return coords[2];
//}
//
//float Vector::GetW() const
//{
//    assert(dim > 3);
//	return coords[3];
//}

//void Vector::Add(const Vector& v)
//{
//    assert(v.GetDim() == dim);
//    for (int i = 0; i < dim; i++) {
//        coords[i] += v.GetCoord(i);
//    }
//}
//
//void Vector::Mul(const float c)
//{
//    for (int i = 0; i < dim; i++) {
//        coords[i] *= c;
//    }
//}
//
//void Vector::Sub(const Vector& v)
//{
//    assert(v.GetDim() == dim);
//    for (int i = 0; i < dim; i++) {
//        coords[i] -= v.GetCoord(i);
//    }
//}

// Calculates the scalar product of the given vector and this vector
float Vector::DotProduct(const Vector& v) const
{
    assert(v.mDim == mDim);
    float product = 0;
    for (int i = 0; i < mDim; i++) {
        product += v.mpCoords[i] * mpCoords[i];
    }
	return product;
}

Vector Vector::CrossProduct(const Vector& v) const
{
    assert(v.mDim == mDim && mDim == 3);
    float vx = mpCoords[1] * v.mpCoords[2] - mpCoords[2] * v.mpCoords[1];
    float vy = mpCoords[2] * v.mpCoords[0] - mpCoords[0] * v.mpCoords[2];
    float vz = mpCoords[0] * v.mpCoords[1] - mpCoords[1] * v.mpCoords[0];
	return Vector(vx, vy, vz);
}

// Returns the dimension of this vector
int Vector::GetDim() const
{
	return mDim;
}

// Returns the norm of this vector
float Vector::Length() const
{
    if (mDim == 1) {
        return fabs(mpCoords[0]);
    }
	return sqrt(DotProduct(*this));
}

// Returns the squared norm of this vector
float Vector::Length2() const
{
	return DotProduct(*this);
}

// Normalizes the vector and returns the reference to this
Vector& Vector::Normalize() {
    *this /= Length();
    return (*this);
}

Vector Vector::GetProjection(const Vector& axis) const {
    Vector projection = Vector(axis).Normalize();
    return projection * projection.DotProduct(*this);
}

string Vector::ToString() const {
	string ret_val = "[";
	for (int i = 0; i < mDim; i++) {
		ret_val += (i > 0 ? " " : "") + to_string(mpCoords[i]);
	}
	ret_val += "]";
	return ret_val;
}
