// Class automatically generated by Dev-C++ New Class wizard

#include <GL/gl.h>

#include "mesh.h" // class's header file
//#include "plane.h"
//#include "projection.h"

using namespace engine3d;

// class constructor
Mesh::Mesh(int type)
{
	this->type = type;
}

// copy constructor
//Vertices::Vertices(const Vertices& vs) : Shape(vs)
//{
//	type = vs.type;
//}

void Mesh::Copy(const Mesh& rMesh)
{
    Shape::Copy(rMesh);
	type = rMesh.type;
//    for (int i = 0; i < v.vertices.size(); i++) {
//        vertices.push_back(v.vertices[i]->Clone());
//    }
}

Mesh* Mesh::Clone()
{
    Debug("Mesh.Clone");
    Mesh* p_m = new Mesh(type);
    p_m->Copy(*this);
    return p_m;
}

// class destructor
Mesh::~Mesh()
{
//	for (int i = 0; i < vertices.size(); i++) {
//        delete vertices[i];
//    }
//    vertices.resize(0);
}

/** 
 * Adds new vertex to the set of vertices.
 * This method clones vertex internally so use it only
 * after all vertex attributes have been set.
 * Otherwise use Node::AddChild
 **/
void Mesh::AddVertex(const Vertex& v)
{
    Vertex* vertex = new Vertex(v);
	//vertices.push_back(vertex);
	AddChild(vertex);
}

/**
 * @param index of the vertex to retrieve
 * @return vertex at the given index
 * If no indices are defined vertices[index] is returned, 
 * otherwise vertices[indices[index]] is returned
 **/
//Vertex& Vertices::GetVertex(int index) const 
//{
//    if (indices.Size() <= 0) {
//        assert(index >= 0 && index < mChildren.Size());
//        Vertex* p_vertex = dynamic_cast<Vertex*>(mChildren[index]);
//        assert(p_vertex);
//        return *p_vertex;
//        //assert(index >= 0 && index < vertices.size());
//        //return *vertices[index];
//    }
//    else {
//        assert(index >= 0 && index < indices.Size());
//        assert(indices[index] < mChildren.Size());
//        Vertex* p_vertex = dynamic_cast<Vertex*>(mChildren[indices[index]]);
//        assert(p_vertex);
//        return *p_vertex;
//        //assert(indices[index] < vertices.size());
//        //return *vertices[indices[index]];
//    }
//}

/**
 * Adds new vertex to the set of vertices.
 **/
void Mesh::AddVertex(const Vector& v, const Color& color)
{
    Vertex* vertex = new Vertex(v);
    vertex->SetColor(color);
	//vertices.push_back(vertex);
	AddChild(vertex);
}

int Mesh::GetSize() const 
{
    return mChildren.Size();
    //return vertices.size();
}

/**
 * Adds new vertex to the set of vertices
 **/
void Mesh::AddVertex(const Vector& v, const Vector& texCoords)
{
    Vertex* vertex = new Vertex(v);
    vertex->SetTexCoords(texCoords);
	//vertices.push_back(vertex);
	AddChild(vertex);
}

void Mesh::Render() {
//	if (distant) {
//		RenderDistant();
//	}
//	else {
//        glBegin(type);
//        Shape::Render();
//        glEnd();
//	}

//	else {
        //Shape::BeginRender();
        glBegin(type);
        Shape::Render();
        glEnd();
        //Shape::EndRender();
//    }
}

// Sets texture coordinates for all vertices
void Mesh::SetTexCoords(List<Vector*>& texCoords)
{
	//assert(texCoords.size() == vertices.size());
	assert(texCoords.Size() == mChildren.Size());
	for (int i = 0; i < mChildren.Size(); i++) {
        //vertices[i]->SetTexCoords(*texCoords[i]);
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

/*
 * Projects the front face of the vertices to the far plane 
 * of the clipping area
 */
Shape* Mesh::GetDistantProjection()
{
//	Vertices distantProjection;
//	distantProjection.SetDistant(true);
//	Plane plane = GetRoot().GetCamera().GetFarPlane();
//	Projection p;
//	p.AddPlane(plane);
//	p.SetDirection(plane.GetNormal());
//	for (int i = 0; i < vertices.size(); i++) {
//        distantProjection.AddVertex(p.Project(vertices[i]->GetCoords()), vertices[i]->GetTexCoords());
//    }
//    
    return NULL;//(Shape) distantProjection;
}