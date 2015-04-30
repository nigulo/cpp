#include "matrix.h"
#include <cassert>

using namespace engine3d;

Matrix::Matrix(int dim)
{
    assert(dim > 0);
	numRows = dim;
	numColumns = dim;
	elements = new float[dim * dim];
}

Matrix::Matrix(int rows, int columns)
{
    assert(rows > 0 && columns > 0);
	numRows = rows;
	numColumns = columns;
	elements = new float[rows * columns];
}

Matrix::Matrix(const Matrix& m) {
    numRows = m.numRows;
    numColumns = m.numColumns;
	elements = new float[numRows * numColumns];
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; j++) {
            Set(i, j, m.Get(i, j));
        }
    }
}

void Matrix::operator=(const Matrix& m) {
    if (elements) {
        delete [](elements);
    }
    numRows = m.numRows;
    numColumns = m.numColumns;
	elements = new float[numRows * numColumns];
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; j++) {
            Set(i, j, m.Get(i, j));
        }
    }
}

Matrix::~Matrix()
{
    if (elements) {
        delete [](elements);
    }
}

void Matrix::SetRow(int index, const Vector& row) {
    assert(index < numRows && row.GetDim() <= numColumns);
    for (int i = 0; i < row.GetDim(); i++) {
        Set(index, i, row[i]);
    }
}

void Matrix::SetColumn(int index, const Vector& column) {
    assert(index < numColumns && column.GetDim() <= numRows);
    for (int i = 0; i < column.GetDim(); i++) {
        Set(i, index, column[i]);
    }
}

Matrix Matrix::operator+(const Matrix& m) const {
    assert(numRows == m.numRows && numColumns == m.numColumns);
    Matrix m1(numRows, numColumns);
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; j++) {
            m1.Set(i, j, Get(i, j) + m.Get(i, j));
        }
    }
    return m1;
}

Matrix Matrix::operator-(const Matrix& m) const {
    assert(numRows == m.numRows && numColumns == m.numColumns);
    Matrix m1(numRows, numColumns);
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; j++) {
            m1.Set(i, j, Get(i, j) - m.Get(i, j));
        }
    }
    return m1;
}

Matrix Matrix::operator*(const Matrix& m) const {
    assert(numColumns == m.numRows && numRows == m.numRows && numColumns == m.numColumns);
    Matrix m1(numRows, m.numColumns);
    for (int k = 0; k < numRows; k++) {
        for (int j = 0; j < m.numColumns; j++) {
            float d = 0;
            for (int l = 0; l < numColumns; l++) {
                d += Get(k, l) * m.Get(l, j);
            }
            m1.Set(k, j, d);
        }
    }
    return m1;
}

Matrix Matrix::operator*(float k) const {
    Matrix m1(numRows, numColumns);
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; j++) {
            m1.Set(i, j, Get(i, j) * k);
        }
    }
    return m1;
}

Vector Matrix::operator*(const Vector& v) const {
    assert(numColumns == v.GetDim());
    Vector v1(numRows);
    for (int k = 0; k < numRows; k++) {
        float d = 0;
        for (int l = 0; l < numColumns; l++) {
            d += Get(k, l) * v[l];
        }
        v1.SetCoord(k, d);
    }
    return v1;
}

Matrix Matrix::GetUnit(int dim) {
    Matrix m(dim, dim);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (i == j) {
                m.Set(i, j, 1);
            }
            else {
                m.Set(i, j, 0);
            }
        }
    }
    return m;
}

void Matrix::Set(int row, int col, float d) {
    assert(row < numRows && col < numColumns);
    elements[col * numRows + row] = d;
}

float Matrix::Get(int row, int col) const {
    assert(row < numRows && col < numColumns);
    return elements[col * numRows + row];
}

const float* Matrix::GetElements() const {
    return elements;
}

Matrix Matrix::Transpose() const {
	Matrix m(numColumns, numRows);
    for (int i = 0; i < numRows; i++) {
        for (int j = i + 1; j < numColumns; j++) {
        	m.Set(j, i, Get(i, j));
        }
    }
    return m;
}
