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
	Perspective(Program& rProgram,
			float viewAngle = 90.0f, float aspect = 1.0f, float near = 0.1f, float far = 100.0f);
	virtual ~Perspective();

	/**
	 * @return view angle in y-direction in degrees
	 **/
	float GetViewAngle() const {
        return mViewAngle;
    }

	/**
	 * @return ratio of x (width) to y (height)
	 **/
	float GetAspect() const {
        return mAspect;
    }

	virtual const Plane GetTopPlane(const Camera& r_camera) const;
    virtual const Plane GetBottomPlane(const Camera& r_camera) const;
    virtual const Plane GetLeftPlane(const Camera& r_camera) const;
    virtual const Plane GetRightPlane(const Camera& r_camera) const;
    virtual void Update(int width, int height);

private:
    void Update();
private:
	/**
	 * View angle in y-direction in degrees
	 **/
	float mViewAngle;

	/**
	 * Ratio of x to y
	 **/
	float mAspect;
};

}
#endif /* SCENEGRAPH_PERSPECTIVE_H_ */
