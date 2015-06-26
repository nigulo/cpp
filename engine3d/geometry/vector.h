#ifndef VECTOR_H
#define VECTOR_H

#include "base/object.h"

using namespace base;

namespace engine3d {

/**
 * Represents geometrical vector
 **/
class Vector : public Object
{
	public:
		explicit Vector(int dim = 3, bool zeroCoords = true);
		Vector(int dim, float coords[]);
		Vector(float x, float y, float z, float w);
		Vector(float x, float y, float z);
		Vector(float x, float y);
		explicit Vector(float x);
		Vector(const Vector& v);
		// class destructor
		virtual ~Vector();
		
		friend bool operator==(const Vector& v1, const Vector& v2) {
            if (v1.mDim != v2.mDim) {
                return false;
            }
            for (int i = 0; i < v1.mDim; i++) {
                if (v1.mpCoords[i] != v2.mpCoords[i]) {
                    return false;
                }
            }
            return true;
        }

		friend bool operator!=(const Vector& v1, const Vector& v2) {
            return !(operator==(v1, v2));
        }
		
		Vector& operator=(const Vector& v);
		Vector operator-() const;
		Vector operator+(const Vector& v) const;
		Vector operator-(const Vector& v) const;
		Vector operator*(float c) const;
		Vector operator/(float c) const;
		Vector& operator+=(const Vector& v);
		Vector& operator-=(const Vector& v);
		Vector& operator*=(float c);
		Vector& operator/=(float c);
		float operator[](int i) const;
		// returns the coordinate of this vector at the specified index
		//float GetCoord(int index) const;
		void SetCoord(int index, float d);
		//float GetX() const;
		//float GetY() const;
		//float GetZ() const;
		//float GetW() const;
		//void Add(const Vector& v);
		//void Sub(const Vector& v);
		//void Mul(const float c);
		
		// Returns the squared norm of this vector
		float Length2() const;
		// Returns the norm of this vector
		float Length() const;
		Vector GetProjection(const Vector& axis) const;
		// Returns the dimention of this vector
		int GetDim() const;
		void Rotate(Vector& v, float angle);
		/**
         * @return the scalar product of this vector and given vector
         **/
		float DotProduct(const Vector& v) const;
		/**
		 * @return *this x v
		 **/
		Vector CrossProduct(const Vector& v) const;
		Vector& Normalize();
		string ToString() const;
		float* GetCoords() {
			return mpCoords;
		}

	protected:
        int mDim;
        float* mpCoords;
};
}
#endif // VECTOR_H
