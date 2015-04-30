#ifndef MATRIX_H
#define MATRIX_H

#include "base/object.h"
#include "vector.h"

using namespace base;

namespace engine3d {
/*
 * Implements matrix algebra
 */
class Matrix
{
	public:
		Matrix(int dim = 3);
		Matrix(int rows, int columns);
		Matrix(const Matrix& m);
		virtual ~Matrix();
		void operator=(const Matrix& m);
		void SetRow(int index, const Vector& row);
		void SetColumn(int index, const Vector& column);
		void Set(int row, int col, float d);
		float Get(int row, int col) const;
		Matrix operator+(const Matrix& m) const;
		Matrix operator-(const Matrix& m) const;
		Matrix operator*(const Matrix& m) const;
		Matrix operator*(float k) const;
		/**
		 * Multiplies the matrix with a column vector
		 */
		Vector operator*(const Vector& v) const;
		Matrix Transpose() const;
		const float* GetElements() const;
		static Matrix GetUnit(int dim = 3);
	protected:
        int numRows;
        int numColumns;
        float* elements;
};
}
#endif // MATRIX_H
