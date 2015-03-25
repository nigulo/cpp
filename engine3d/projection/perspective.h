/*
 * Perspective.h
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#ifndef SCENEGRAPH_PERSPECTIVE_H_
#define SCENEGRAPH_PERSPECTIVE_H_

#include "projection.h"

namespace engine3d {

class Camera;
class Perspective : public Projection {
public:
	Perspective(double viewAngle = 90.0f, double aspect = 1.0f, double near = 0.5f, double far = 10.0f);
	virtual ~Perspective();

	virtual void Init() const;

	/**
	 * @return view angle in y-direction in degrees
	 **/
	double GetViewAngle() const {
        return mViewAngle;
    }

	/**
	 * @return ratio of x (width) to y (height)
	 **/
	double GetAspect() const {
        return mAspect;
    }

	virtual const Plane GetTopPlane(const Camera& r_camera) const;
    virtual const Plane GetBottomPlane(const Camera& r_camera) const;
    virtual const Plane GetLeftPlane(const Camera& r_camera) const;
    virtual const Plane GetRightPlane(const Camera& r_camera) const;

private:
	/**
	 * View angle in y-direction in degrees
	 **/
	double mViewAngle;

	/**
	 * Ratio of x to y
	 **/
	double mAspect;
};

}
#endif /* SCENEGRAPH_PERSPECTIVE_H_ */
