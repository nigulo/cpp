#include "trianglestrip.h" // class's header file

#include "engine3d/geometry/triangle.h"
#include "engine3d/scenegraph/scene.h"
#include "engine3d/scenegraph/camera.h"
#include <GL/glew.h>

using namespace engine3d;
// class constructor
TriangleStrip::TriangleStrip() : Mesh(GL_TRIANGLE_STRIP)
{
}

void TriangleStrip::Copy(const TriangleStrip& ts)
{
    Mesh::Copy(ts);
}

TriangleStrip* TriangleStrip::Clone() const
{
    Debug("TriangleStrip.Clone");
    TriangleStrip* p_ts = new TriangleStrip();
    p_ts->Copy(*this);
    return p_ts;
}

// class destructor
TriangleStrip::~TriangleStrip()
{
	// insert your code here
}

void TriangleStrip::Render() 
{
    Mesh::Render();
}

