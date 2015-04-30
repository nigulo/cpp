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
    		 * Constructs a spatial object
    		 **/
    		Spatial();
    		
    		/**
    		 * Constructs a spatial object with the given name
    		 **/
    		Spatial(const string& rName);
    		
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
    		virtual void SetTransformation(const Transformation& rT);
    		/**
    		 * @return the local transformation for this object.
    		 **/
    		virtual const Transformation& GetTransformation() const;
    		/**
    		 * @return the world transformation for this object.
    		 * World transformation is calculated from local 
             * transformation and parent node's world transformation.
    		 **/
            virtual void Transform();
    
            /**
             * @return true if this object has moved or rotated
             * in space after the scene was rendered last time
             **/		
    		virtual bool IsChanged() {return mChanged;}
    		
    	protected:
            
            /**
             * Current local transformation of the object
             **/
            Transformation mTransformation;
            
            /**
             * New local transformation of the object
             **/
            Transformation mNewTransformation;
            
            /**
             * Specified if this object has moved or rotated
             * after the scene was rendered last time
             **/
            bool mChanged;
            
    };
}
#endif // SPATIAL_H
