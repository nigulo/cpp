#ifndef PARTICLE_H
#define PARTICLE_H

#include "engine3d/meshes/vertex.h"

namespace engine3d {
    /**
     * Class cointaining physical properties of a point
     * mass body.
     */
    class Particle : public Vertex
    {
    	public:
    		Particle();
    		~Particle();
    	protected:
            double mMass;
            Vector mVelocity;
            
    };
}
#endif // PARTICLE_H
