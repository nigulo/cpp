/*
 * Projection.h
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#ifndef PROJECTION_PROJECTION_H_
#define PROJECTION_PROJECTION_H_

#include "base/object.h"
#include "engine3d/geometry/plane.h"

using namespace base;

namespace engine3d {

class Camera;
class Projection : public Object {

protected:
	Projection(double near = 0.5f, double far = 10.0f);
	virtual ~Projection();

public:
	virtual void Init() const;
    /**
     * @return disctance of the near clipping plane measured from eye point
     **/
	double GetZNear() const {
        return mZNear;
    }

    /**
     * @return disctance of the far clipping plane measured from eye point
     **/
	double GetZFar() const {
        return mZFar;
    }

	virtual const Plane GetTopPlane(const Camera& r_camera) const = 0;
    virtual const Plane GetBottomPlane(const Camera& r_camera) const = 0;
    virtual const Plane GetLeftPlane(const Camera& r_camera) const = 0;
    virtual const Plane GetRightPlane(const Camera& r_camera) const = 0;

protected:

    /**
     * Disctance of the near clipping plane measured from eye point
     **/
    double mZNear;

    /**
     * Disctance of the far clipping plane measured from eye point
     **/
    double mZFar;

};

}
#endif /* PROJECTION_PROJECTION_H_ */
