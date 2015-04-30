#ifndef PARAMETRICCURVE_H
#define PARAMETRICCURVE_H

#include "base/object.h"
#include "vector.h"
#include <vector>

using namespace std;

namespace engine3d {
    class ParametricCurve
    {
    	public:
    		ParametricCurve();
    		~ParametricCurve();
    		
    		/**
    		 * Sets the array of control points which define the curve.
    		 **/
    		void SetPoints(const vector<Vector>& rPoints);
    		
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
    		vector<Vector> mPoints;
    	
    };
}
#endif // PARAMETRICCURVE_H
