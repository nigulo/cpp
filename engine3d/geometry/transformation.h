#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include "base/object.h"
#include "vector.h"
#include "matrix.h"

using namespace base;

namespace engine3d {
/*
 * Implements spatial transformations
 */
class Transformation
{
	public:
        /**
		 * Creates a unit transformation
		 **/
		Transformation();
		Transformation(const Transformation& t);
		Transformation(const Matrix& m);
		Transformation(const Vector& translation);
		Transformation(const Vector& rotAxis, float rotAngle);
		Transformation(const Vector& rotAxis, float rotAngle, const Vector& translation);
        Transformation& operator=(const Transformation& t);
        // class destructor
		~Transformation();

		void SetTranslation(const Vector& shift);

		/**
		 * Rotates around the given axes
		 **/
		void SetRotation(const Vector& axis, float angle);
		void SetMatrix(const Matrix& m);
		const Matrix& GetMatrix() const;
		/**
		 * @return the translation component of the current transformation
		 **/
		Transformation GetTranslation() const;
		
		/**
		 * @return the rotation component of the current transformation
		 **/
		Transformation GetRotation() const;

		Transformation operator*(const Transformation&) const;

        /**
         * Performs the OpenGL transformation
         **/
		void Transform() const;
		
		/**
		 * Transforms the given vector
		 * @return vector transformed by this transformation
		 **/
		Vector Transform(const Vector& v) const;
		
		/**
         * Translates the given vector (rotation component 
         * of the transformation is disregarded).
         * @return vector transformed by this transformation
		 **/
		Vector Translate(const Vector& v) const;

		/**
		 * Rotates the given vector (translation component of the
         * transformation disregarded).
         * @return vector transformed by this transformation
		 **/
		Vector Rotate(const Vector& v) const;
		
	private:
        
        /**
         * Transformation matrix (contains both 
         * rotation and translation components).
         **/
        Matrix mMatrix;
};
}
#endif // TRANSFORMATION_H
