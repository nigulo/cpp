#include "triangles.h"

#include "engine3d/geometry/triangle.h"
#include "engine3d/scenegraph/scene.h"
#include "engine3d/scenegraph/camera.h"
#include <GL/gl.h>

using namespace engine3d;
// class constructor
Triangles::Triangles() : Mesh(GL_TRIANGLES)
{
}

void Triangles::Copy(const Triangles& ts)
{
    Mesh::Copy(ts);
}

Triangles* Triangles::Clone() const
{
    Debug("Triangles.Clone");
    Triangles* p_ts = new Triangles();
    p_ts->Copy(*this);
    return p_ts;
}


// class destructor
Triangles::~Triangles()
{
}


// Adds new triangle
void Triangles::Add(const Vertex& v1, const Vertex& v2, const Vertex& v3)
{
    AddVertex(v1);
    AddVertex(v2);
    AddVertex(v3);
}

void Triangles::Render() {
    Mesh::Render();
}

