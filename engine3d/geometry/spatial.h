#ifndef SPATIAL_H
#define SPATIAL_H

#include "base/object.h"
#include "transformation.h"

using namespace base;

namespace engine3d {
    /**
     * Base class for all objects that can be transformed in space.
     **/
    class Spatial : public Object
    {
    	protected:
    		/**
    		 * Constructs a spatial object with the given name
    		 **/
    		Spatial(const string& rName = "");
    		
    		/**
    		 * Copy constructor
    		 **/
    		Spatial(const Spatial& rSpatial);
    		
    		/**
    		 * Copies the given spatial object into this
    		 * object.
    		 **/
    		void Copy(const Spatial& rSpatial);
    		virtual ~Spatial();
    		
    	public:	
            /**
             * Sets the local transformation for this object.
             **/
    		void SetTransformation(const Transformation& rT);
    		/**
    		 * @return the local transformation for this object.
    		 **/
    		const Transformation& GetTransformation() const;

    		/**
    		 * @return the world transformation for this object.
    		 **/
    		const Transformation& GetWorldTransformation() const;

    		/**
    		 * Replaces the current transformation with the new one
    		 **/
            virtual void Transform();

    		/**
    		 * Reverts the new transformation back to old one
    		 **/
            void Revert();

            /**
             * @return true if this object has moved or rotated
             * in space after the scene was rendered last time
             **/		
    		bool IsChanged() {
    			return mChanged;
    		}

    		void Reset() {
    			mChanged = false;
    		}
    		
    		const Vector& GetPosition() const {
    			return mPosition;
    		}

    	protected:
    		/**
    		 * @return the new local transformation for this object.
    		 **/
    		const Transformation& GetOldTransformation() const;
    		/**
    		 * @return the new world transformation for this object.
    		 **/
    		const Transformation& GetOldWorldTransformation() const;
            /**
             * Sets the local and world transformations for this object.
             **/
    		void SetWorldTransformation(const Transformation& rT);
    	private:
            
            /**
             * Current local transformation of the object
             **/
            Transformation mOldTransformation;
            
            /**
             * New local transformation of the object
             **/
            Transformation mTransformation;

            /**
             * Current world transformation of the object
             **/
            Transformation mOldWorldTransformation;

            /**
             * New world transformation of the object
             **/
            Transformation mWorldTransformation;
            
            /**
             * Specified if this object has moved or rotated
             * after the scene was rendered last time
             **/
            bool mChanged;

            /**
             * Position of point-like object with given transformation.
             */
            Vector mPosition;
            
    };
}
#endif // SPATIAL_H
