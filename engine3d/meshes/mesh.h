#ifndef MESH_H
#define MESH_H

#include "shape.h"
#include "vertex.h"
#include "engine3d/geometry/vector.h"
#include "engine3d/program/vertexbuffer.h"
#include "engine3d/program/elementbuffer.h"

#include <vector>
#include <memory>

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
		Vertex& AddVertex(const Vertex& vertex);
		Vertex& AddVertex(const Vector& v, const Color& color);
		Vertex& AddVertex(const Vector& v, const Vector& texCoords);
		// Adds a new child index
		void AddIndex(int index);
		// Adds a set of child indices
		void AddIndices(const int* indices, int count);
		// Returns the vertex at the given index
		//Vertex& GetVertex(int index) const;
		int GetSize() const;
		virtual void Update();
		virtual void Render();
		// Generates texture coordinates for the vertices
		virtual void GenTexCoords();
		void SetTexCoords(vector<Vector*>& texCoords);
	protected:
	protected:
		int mMode;
		unique_ptr<VertexBuffer> mpVertexBuffer;
		unique_ptr<ElementBuffer> mpElementBuffer;
		vector<Vertex*> mVertices;
        vector<unsigned short> mIndices;
};
}
#endif // MESH_H
