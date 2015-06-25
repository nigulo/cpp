#include "mesh.h"
#include "engine3d/scenegraph/scene.h"
#include <cassert>
#include <GL/glew.h>

using namespace engine3d;

Mesh::Mesh(int mode) :
		mMode(mode),
		mpVertexBuffer(nullptr),
		mpElementBuffer(nullptr) {
}

void Mesh::Copy(const Mesh& rMesh)
{
    Shape::Copy(rMesh);
    for (auto&& v : rMesh.mVertices) {
    	mVertices.push_back(new Vertex(*v));
    }
    for (auto&& i : rMesh.mIndices) {
        mIndices.push_back(i);
    }
}

Mesh* Mesh::Clone()
{
    Debug("Mesh.Clone");
    Mesh* p_m = new Mesh(mMode);
    p_m->Copy(*this);
    return p_m;
}

Mesh::~Mesh()
{
    for (auto i = mVertices.begin(); i != mVertices.end(); i++) {
    	assert(*i);
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
Vertex& Mesh::AddVertex(const Vertex& v)
{
    Vertex* p_vertex = new Vertex(v);
	mVertices.push_back(p_vertex);
	return *p_vertex;
}

/**
 * Adds new vertex to the set of vertices.
 **/
Vertex& Mesh::AddVertex(const Vector& v, const Color& color)
{
    Vertex* p_vertex = new Vertex(v);
    p_vertex->SetColor(color);
	AddVertex(*p_vertex);
	return *p_vertex;
}

/**
 * Adds new vertex to the set of vertices
 **/
Vertex& Mesh::AddVertex(const Vector& v, const Vector& texCoords)
{
    Vertex* p_vertex = new Vertex(v);
    p_vertex->SetTexCoords(texCoords);
	AddVertex(*p_vertex);
	return *p_vertex;
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
    return mVertices.size();
}


void Mesh::Update() {
	Shape::Update();
	Debug("Mesh::Update 1");
	if (!mpVertexBuffer) {
		mpVertexBuffer.reset(new VertexBuffer(GetScene().GetProgram()));
	}
	if (!mpElementBuffer) {
		mpElementBuffer.reset(new ElementBuffer(mMode));
	}
	Debug("Mesh::Update 2");
	UpdateVertices();
	Debug("Mesh::Update 3");
	mpVertexBuffer->SetData(mVertices);
	Debug("Mesh::Update 3.5");
	mpElementBuffer->SetData(mIndices);
	Debug("Mesh::Update 4");
}

void Mesh::UpdateVertices() {
	Transformation wt = GetWorldTransformation();
	for (auto&& p_vertex : mVertices) {
		p_vertex->SetCoords(wt.Transform(p_vertex->GetCoords()));
	}
}

void Mesh::Render() {
	Debug("Mesh::Render 1");
	Shape::Render();
	Debug("Mesh::Render 2");
	mpVertexBuffer->Enable();
	mpVertexBuffer->Render();
	mpElementBuffer->Render();
	mpVertexBuffer->Disable();
}

// Sets texture coordinates for all vertices
void Mesh::SetTexCoords(vector<Vector*>& texCoords)
{
	assert(texCoords.size() == mVertices.size());
	for (unsigned i = 0; i < mVertices.size(); i++) {
        mVertices[i]->SetTexCoords(*texCoords[i]);
    }
}

// Generates texture coordinates for the vertices
// Implement this method in sub-classes
void Mesh::GenTexCoords()
{
}


