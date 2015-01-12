// Class automatically generated by Dev-C++ New Class wizard

#ifndef PARAMETRICCURVE_H
#define PARAMETRICCURVE_H

#include "base/object.h" // inheriting class's header file
#include "vector.h"
#include "base/arraylist.h"

namespace engine3d {
    /**
     * No description
     */
    class ParametricCurve : public Object
    {
    	public:
    		// class constructor
    		ParametricCurve();
    		// class destructor
    		~ParametricCurve();
    		
    		/**
    		 * Sets the array of control points which define the curve.
    		 **/
    		void SetPoints(const ArrayList<Vector>& rPoints);
    		
    		/**
    		 * @return a curve point for the given parameter value.
    		 **/
    		virtual Vector GetPoint(double parameter) const = 0;

    		/**
    		 * @return minimum allowed parameter value (including)
    		 **/
    		virtual double GetParameterMin() const {
                return 0;
            }

    		/**
    		 * @return maximum allowed parameter value (excluding)
    		 **/
    		virtual double GetParameterMax() const {
                return 1;
            }
            
    	protected:
            ArrayList<Vector> mPoints;
    	
    };
}
#endif // PARAMETRICCURVE_H
