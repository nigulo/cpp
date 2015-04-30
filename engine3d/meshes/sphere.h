#ifndef SPHERE_H
#define SPHERE_H

#include "trianglestrip.h"
#include "triangles.h"
#include <vector>

using namespace std;

namespace engine3d {
/*
 * Defines a geometrical sphere
 */
class Sphere : public Shape
{
	public:
		// class constructor
		Sphere(float radius, int parts1 = 10, int parts2 = 10, bool genTexCoords = true, float completeness = 1.0);
	protected:
        void Copy(const Sphere& ts);
    public:
        virtual Sphere* Clone() const;
		// class destructor
		virtual ~Sphere();
		virtual void Render();
	protected:
        float radius;
        int parts1;
        int parts2;
        /**
         * Specifies the completeness level of sphere
         * If completeness is:
         * 1 - whole sphere is created
         * 0.5 - half sphere is created
         * All completeness levels from 0 to 1 are supported
         **/ 
        float completeness;
        //----------------------------
        Triangles* top;
        vector<TriangleStrip*> rings;
        Triangles* bottom;
        //----------------------------
};
}
#endif // SPHERE_H
