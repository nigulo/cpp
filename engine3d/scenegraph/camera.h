#ifndef CAMERA_H
#define CAMERA_H

#include "node.h"
#include "engine3d/geometry/vector.h"
#include "engine3d/geometry/matrix.h"
#include "engine3d/geometry/plane.h"
#include "engine3d/geometry/transformation.h"
#include "engine3d/containment/boundingvolume.h"
#include "engine3d/projection/projection.h"
#include <memory>

namespace engine3d {
    
class Camera : public Node
{
	public:
		Camera(Program& rProgram, Projection* pProjection);
		
		virtual ~Camera();
		
		virtual void Render() {}
		void Look();
		
		/**
		 * @return the eye position
		 **/
		const Vector& GetEye() const {
            return mEye;
        }
        
		/**
		 * @deprecated DONT USE THIS METHOD. 
         * USE GetDirection instead which returns normalized
         * vector pointing towards the center of the scenr
		 **/
		const Vector& GetCenter() const {
            return mCenter;
        }
        
		/**
		 * @return the vector pointing up
		 **/
		const Vector& GetUp() const {
            return mUp;
        }
        
		/**
		 * @return the vector pointing to the right
		 **/
		const Vector GetRight() const {
            return GetDirection().CrossProduct(mUp);
        }
        
		/**
		 * @return normalized vector pointing towards the center 
         * of the scene
		 **/
		const Vector GetDirection() const {
            return (mCenter - mEye).Normalize();
        }
		
        const Plane GetFarPlane() const;
        const Plane GetNearPlane() const;
        const Plane GetTopPlane() const;
        const Plane GetBottomPlane() const;
        const Plane GetLeftPlane() const;
        const Plane GetRightPlane() const;

        /**
         * Sets the camera's world transformation. Transformation
         * defines the location and direction of the camera.
         * @param rTransformation new transformation of the camera. 
         **/
        //void SetTransformation(const Transformation& rTransformation);

        /**
         * Checks if the given volume is out of the camera's view.
         * @rVolume bounding volume to check for culling
         * @return ture if the given volume can be culled, false otherwise
         **/
        bool Cull(const BoundingVolume& rVolume);
        
        void Transform();
        void Update();
        
        const Projection& GetProjection() const {
        	return *mpProjection;
        }

	private:
        
        /**
         * Eye point of the camera
         **/
        Vector mEye;
        
        /**
         * Vector defining the point where camera is directed
         **/
        Vector mCenter;
        
        /**
         * Up vector of the camera
         **/
        Vector mUp;
        
        /**
         * Specifies if perspective or parallel projection is used
         */
        unique_ptr<Projection> mpProjection;
        
        const Uniform& mrUniform;
        Matrix mMatrix;
        
};
}
#endif // CAMERA_H
