#ifndef LINE_H
#define LINE_H

#include "base/object.h"
#include "vector.h"
#include <memory>

using namespace base;

namespace engine3d {

    class Segment;
        
    class Line
    {
    	public:
    		Line(const Vector& rPoint1, const Vector& rPoint2);
    		virtual ~Line();
    		
            virtual bool Contains(const Vector& rPoint) const;
    		
    		/**
    		 * @return the distance between given point and this line
    		 **/
    		double GetDistance(const Vector& rPoint) const;
    		/**
    		 * @return the distance vector from the line to the point
    		 **/
    		Vector GetDistanceVector(const Vector& rPoint) const;
    		
    		/**
    		 * @return squared minimum distance between two lines
    		 **/
    		double GetSquaredDistance(const Line& rLine) const;
    		
    		/**
    		 * @return pointer to segment whose first point lies
    		 * on this line and second point on other line
    		 **/
    		const unique_ptr<Segment> GetDistance(const Line& rLine) const;
    		
    		virtual bool IsSegment() const {
                return false;
            }
            
            virtual bool Crosses(const Line& rLine) const;
    		
    		string ToString() const;

    	public:
            const Vector mPoint1;
            const Vector mPoint2;
            /**
             * mPoint2 - mPoint1
             **/
            const Vector mDirection;
    };
}
#endif // LINE_H
