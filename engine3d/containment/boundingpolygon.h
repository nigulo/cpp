#ifndef BOUNDINGPOLYGON_H
#define BOUNDINGPOLYGON_H

#include "boundingvolume.h"
#include "engine3d/geometry/vector.h"
#include <vector>

using namespace std;

namespace engine3d {
    /**
     * This class represents the convex polygon for
     * model bounding 
     **/
    class BoundingPolygon : public BoundingVolume
    {
    	public:
    		BoundingPolygon() {}
    		/**
    		 * Constructs bounding polygon from 
    		 * the given array of vertices
    		 **/
    		BoundingPolygon(Vector vertices[], int number);
    		
    		BoundingPolygon* Clone() const;
    		
    		~BoundingPolygon();
    		
    		Vector& AddVertex(const Vector& rVertex);

		    int WhichSide(const Plane& rPlane) const;

		    unique_ptr<Vector> Intersects(const Line& rLine) const;
		
    		unique_ptr<Vector> Collides(const BoundingVolume& rOtherBound) const;
    		
    		void Transform();
		private:
            vector<Vector> mVertices;
            vector<Vector> mTransformedVertices;
    };
}

#endif // BOUNDINGPOLYGON_H
