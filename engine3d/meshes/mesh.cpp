#include <GL/gl.h>

#include "mesh.h"

using namespace engine3d;

// class constructor
Mesh::Mesh(int mode) : mElementBuffer(mode) {
}

void Mesh::Copy(const Mesh& rMesh)
{
    Shape::Copy(rMesh);
    for (auto i = rMesh.mIndices.begin(); i != rMesh.mIndices.end(); i++) {
        mIndices.push_back(*i);
    }
}

Mesh* Mesh::Clone()
{
    Debug("Mesh.Clone");
    Mesh* p_m = new Mesh(mElementBuffer.GetMode());
    p_m->Copy(*this);
    return p_m;
}

// class destructor
Mesh::~Mesh()
{
    for (auto i = mVertices.begin(); i != mVertices.end(); i++) {
    	delete (*i);
    }
    mVertices.clear();
    mIndices.clear();
}

/** 
 * Adds new vertex to the set of vertices.
 * This method clones vertex internally so use it only
 * after all vertex attributes have been set.
 * Otherwise use Node::AddChild
 **/
void Mesh::AddVertex(const Vertex& v)
{
    Vertex* p_vertex = new Vertex(v);
	mVertices.push_back(p_vertex);
}

/**
 * Adds new vertex to the set of vertices.
 **/
void Mesh::AddVertex(const Vector& v, const Color& color)
{
    Vertex* p_vertex = new Vertex(v);
    p_vertex->SetColor(color);
	AddVertex(*p_vertex);
}

/**
 * Adds new vertex to the set of vertices
 **/
void Mesh::AddVertex(const Vector& v, const Vector& texCoords)
{
    Vertex* p_vertex = new Vertex(v);
    p_vertex->SetTexCoords(texCoords);
	AddVertex(*p_vertex);
}

void Mesh::AddIndex(int index)
{
    assert(index >= 0);
	mIndices.push_back(index);
}

void Mesh::AddIndices(const int* indices, int count)
{
    for (int i = 0; i < count; i++) {
        assert(indices[i] >= 0);
    	this->mIndices.push_back(indices[i]);
    }
}

int Mesh::GetSize() const
{
    return mChildren.size();
}


void Mesh::UpdateBuffers() {
	mVertexBuffer.SetData(mVertices);
	mElementBuffer.SetData(mIndices);
}

void Mesh::Render() {
	mVertexBuffer.Render();
	mElementBuffer.Render();
}

// Sets texture coordinates for all vertices
void Mesh::SetTexCoords(vector<Vector*>& texCoords)
{
	assert(texCoords.size() == mChildren.size());
	for (unsigned i = 0; i < mChildren.size(); i++) {
        Vertex* p_child = dynamic_cast<Vertex*>(&GetChild(i));
        assert(p_child);
        p_child->SetTexCoords(*texCoords[i]);
    }
}

// Generates texture coordinates for the vertices
// Implement this method in sub-classes
void Mesh::GenTexCoords()
{
}


