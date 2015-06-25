#ifndef BODY_H
#define BODY_H

#include "engine3d/geometry/spatial.h"
#include "engine3d/geometry/vector.h"

namespace engine3d {
    /**
     * Class containing physical properties of body
     */
    class Body : public Object
    {
    	public:
    		Body(Spatial& rSpatial, float mass = 0);
    		~Body();
    		
    		/**
    		 * Applies an external force to body
    		 **/
    		void SetForce(const Vector& rForce);

            const Vector& GetForce() const {
                return mForce;
            }
    		
    		void SetVelocity(const Vector& rVelocity) {
                mVelocity = rVelocity;
            }

            const Vector& GetVelocity() const {
                return mVelocity;
            }

    		/**
    		 * Moves the body according to external
    		 * forces during the given time span.
    		 **/
    		void Move(double dt);
    		
    		// @Override
    		//void CollisionWith(const Node& rNode);

    		const Vector& GetPosition() const {
    			return mrSpatial.GetPosition();
    		}

    		float GetMass() const {
    			return mMass;
    		}

    		//Vector GetForceBetweenBodies() const;
            
    	protected:
    		Spatial& mrSpatial;
            /**
             * Mass of the body
             **/
            float mMass;
            
            /**
             * Velocity of the body
             **/
            Vector mVelocity;
            
            /**
             * External forces applied to body
             **/
            Vector mForce;
    };
}
#endif // BODY_H
