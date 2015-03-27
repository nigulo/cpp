#ifndef TRIANGLEFAN_H
#define TRIANGLEFAN_H

#include "mesh.h" // inheriting class's header file

using namespace base;

namespace engine3d {

class TriangleFan : public Mesh
{
	public:
		// class constructor
		TriangleFan();
	protected:
        void Copy(const TriangleFan& ts);
    public:
        virtual TriangleFan* Clone() const;
		// class destructor
		~TriangleFan();
		virtual void Render();
};
}
#endif // TRIANGLEFAN_H
