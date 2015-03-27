#include <GL/gl.h>

#include "trianglefan.h" // class's header file

#include "engine3d/geometry/triangle.h"

using namespace engine3d;
// class constructor
TriangleFan::TriangleFan() : Mesh(GL_TRIANGLE_FAN)
{
}

void TriangleFan::Copy(const TriangleFan& tf)
{
    Mesh::Copy(tf);
}

TriangleFan* TriangleFan::Clone() const
{
    Debug("TriangleFan.Clone");
    TriangleFan* p_tf = new TriangleFan();
    p_tf->Copy(*this);
    return p_tf;
}

// class destructor
TriangleFan::~TriangleFan()
{
}

void TriangleFan::Render() 
{
    Mesh::Render();
}

