#ifndef MESH_H
#define MESH_H

#include "shape.h"
#include "vertex.h"
#include "engine3d/geometry/vector.h"
#include "engine3d/program/vertexbuffer.h"
#include "engine3d/program/elementbuffer.h"

#include <vector>

using namespace base;
using namespace std;

namespace engine3d {
/*
 * Defines a set of vertices
 * This class should be renamed to Mesh
 */
class Mesh : public Shape
{
	public:
		// class constructor
		Mesh(int type);
	protected:
		void Copy(const Mesh& rMesh);
	public:
		virtual Mesh* Clone();
		// class destructor
		virtual ~Mesh();
		// Adds a vertex to the mesh
		void AddVertex(const Vertex& vertex);
		void AddVertex(const Vector& v, const Color& color);
		void AddVertex(const Vector& v, const Vector& texCoords);
		// Returns the vertex at the given index
		//Vertex& GetVertex(int index) const;
		int GetSize() const;
		virtual void Render();
		// Generates texture coordinates for the vertices
		virtual void GenTexCoords();
		void SetTexCoords(vector<Vector*>& texCoords);
		void UpdateBuffers();
	protected:
	protected:
		VertexBuffer mVertexBuffer;
		ElementBuffer mElementBuffer;
};
}
#endif // MESH_H
