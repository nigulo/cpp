#ifndef TRIANGLES_H
#define TRIANGLES_H

#include "vertex.h"
#include "mesh.h"

using namespace base;

namespace engine3d {

class Triangles : public Mesh
{
	public:
		// class constructor
		Triangles();
		//Triangles(const Triangles& tr);
	protected:
        void Copy(const Triangles& ts);
    public:
        virtual Triangles* Clone() const;
		// class destructor
		virtual ~Triangles();
		void Add(const Vertex& v1, const Vertex& v2, const Vertex& v3);
		virtual void Render();
    protected:
};
}
#endif // TRIANGLES_H
