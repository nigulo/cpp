#include <GL/gl.h>
#include <math.h>

#include "transformation.h" // class's header file

using namespace engine3d;

// class constructor
Transformation::Transformation() : mMatrix(Matrix::GetUnit(4))
{
}

Transformation::Transformation(const Matrix& m) : mMatrix(m)
{
}

// class constructor
Transformation::Transformation(const Transformation& t) : mMatrix(t.mMatrix)
{
}

Transformation::Transformation(const Vector& translation) :
    mMatrix(Matrix::GetUnit(4))
{
    SetTranslation(translation);
}

Transformation::Transformation(const Vector& rotAxis, double rotAngle) :
    mMatrix(Matrix::GetUnit(4))
{
    SetRotation(rotAxis, rotAngle);
}

Transformation::Transformation(const Vector& rotAxis, double rotAngle, const Vector& translation)  :
    mMatrix(Matrix::GetUnit(4))
{
    SetRotation(rotAxis, rotAngle);
    SetTranslation(translation);
}

Transformation& Transformation::operator=(const Transformation& t) 
{
    if (this != &t) {
        mMatrix = t.mMatrix;
    }
    return *this;
}

// class destructor
Transformation::~Transformation()
{
}

// Rotates around the given axes
void Transformation::SetRotation(const Vector& axis, double angle)
{
    Vector a(axis);
    a.Normalize();
    double x = a[0];
    double y = a[1];
    double z = a[2];
    double s = sin(angle);
    double c = cos(angle);
    double c1 = 1 - c;
    double xs = x * s;
    double ys = y * s;
    double zs = z * s;
    double xy = x * y;
    double yz = y * z;
    double xz = x * z;
    mMatrix = Matrix::GetUnit(4);
    mMatrix.Set(0, 0, x * x * c1 + c); mMatrix.Set(0, 1, xy * c1 - zs);   mMatrix.Set(0, 2, xz * c1 + ys);
    mMatrix.Set(1, 0, xy * c1 + zs);   mMatrix.Set(1, 1, y * y * c1 + c); mMatrix.Set(1, 2, yz * c1 - xs);
    mMatrix.Set(2, 0, xz * c1 - ys);   mMatrix.Set(2, 1, yz * c1 + xs);   mMatrix.Set(2, 2, z * z * c1 + c);
}

void Transformation::SetTranslation(const Vector& shift)
{
    mMatrix.Set(0, 3, shift[0]);
    mMatrix.Set(1, 3, shift[1]);
    mMatrix.Set(2, 3, shift[2]);
}

void Transformation::SetMatrix(const Matrix& m) {
    mMatrix = m;
}

const Matrix& Transformation::GetMatrix() const {
    return mMatrix;
}

/**
 * Performs the OpenGL transformation
 **/
void Transformation::Transform() const
{
    
    const double* trMatrix = mMatrix.GetElements();
    glMultMatrixd(trMatrix);
}

Transformation Transformation::operator*(const Transformation& t) const
{
    return Transformation(mMatrix * t.mMatrix);
}

Vector Transformation::Transform(const Vector& v) const
{
    assert(v.GetDim() == 3);
    Vector v1(v[0], v[1], v[2], 1);
    Vector v2 = mMatrix * v1;
    return Vector(v2[0], v2[1], v2[2]);
}

Vector Transformation::Rotate(const Vector& v) const
{
    assert(v.GetDim() == 3);
    Vector v1(v[0], v[1], v[2], 1);
    Matrix m = Matrix::GetUnit(4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m.Set(i, j, mMatrix.Get(i, j));
        }
    }
    Vector v2 = m * v1;
    return Vector(v2[0], v2[1], v2[2]);
}

Vector Transformation::Translate(const Vector& v) const
{
    assert(v.GetDim() == 3);
    Vector v1(v[0], v[1], v[2], 1);
    Matrix m = Matrix::GetUnit(4);
    m.Set(0, 3, mMatrix.Get(0, 3));
    m.Set(1, 3, mMatrix.Get(1, 3));
    m.Set(2, 3, mMatrix.Get(2, 3));
    Vector v2 = m * v1;
    return Vector(v2[0], v2[1], v2[2]);
}

Transformation Transformation::GetTranslation() const
{
    Matrix m = Matrix::GetUnit(4);
    m.Set(0, 3, mMatrix.Get(0, 3));
    m.Set(1, 3, mMatrix.Get(1, 3));
    m.Set(2, 3, mMatrix.Get(2, 3));
    return Transformation(m);
}

Transformation Transformation::GetRotation() const
{
    Matrix m = Matrix::GetUnit(4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m.Set(i, j, mMatrix.Get(i, j));
        }
    }
    return Transformation(m);
}
