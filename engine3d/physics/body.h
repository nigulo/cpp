#ifndef BODY_H
#define BODY_H

#include "engine3d/scenegraph/node.h"
#include "engine3d/scenegraph/collisionlistener.h"
#include "engine3d/geometry/vector.h"

namespace engine3d {
    /**
     * Class containing physical properties of body
     */
    class Body : public Object, CollisionListener
    {
    	public:
    		Body(Node& rNode, float mass = 0, const Vector& rVelocity = Vector());
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
    		void Move(float dt);
    		
    		// @Override
    		void Collision(const Node& rNode1, const Node& rNode2, const unique_ptr<Vector>& rPoint);

    		const Vector& GetPosition() const {
    			return mrNode.GetPosition();
    		}

    		float GetMass() const {
    			return mMass;
    		}

    		//Vector GetForceBetweenBodies() const;
            
    	protected:
    		Node& mrNode;
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
