#ifndef TRIANGLESTRIP_H
#define TRIANGLESTRIP_H

#include "mesh.h"

using namespace base;

namespace engine3d {

class TriangleStrip : public Mesh
{
	public:
		// class constructor
		TriangleStrip();
	protected:
        void Copy(const TriangleStrip& ts);
    public:
        virtual TriangleStrip* Clone() const;
		// class destructor
		virtual ~TriangleStrip();
		virtual void Render();
		
};
}
#endif // TRIANGLESTRIP_H
