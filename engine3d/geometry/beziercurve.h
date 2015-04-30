#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include "parametriccurve.h"

namespace engine3d {
    class BezierCurve : public ParametricCurve
    {
    	public:
    		/**
    		 * @param polynomial order order of the curve. The whole curve is 
    		 * put together from a number of small sections which have the
    		 * given polynomial order. The number of points that define the
             * curve must be multiple of (order + 1). To achieve first order 
             * smoothness all three points at indexes 
             * n * order - 1, n * order and n * order + 1, (where n >= 1)
             * must lie on the same line.
    		 **/
    		BezierCurve(unsigned int order = 2);
    		// class destructor
    		~BezierCurve();
    		/**
    		 * The parameter must be in range 0 <= parameter < 1
    		 **/
    		Vector GetPoint(double parameter) const;

            
    	protected:
            unsigned int mOrder; // actually not used
    };
}
#endif // BEZIERCURVE_H
