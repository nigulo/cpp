/*
 * ortho.h
 *
 *  Created on: Mar 25, 2015
 *      Author: nigul
 */

#ifndef PROJECTION_ORTHO_H_
#define PROJECTION_ORTHO_H_

#include "projection.h"

namespace engine3d {

class Camera;
class Ortho : public Projection {
public:
	Ortho(Program& rProgram,
			float left, float right, float bottom, float top, float near, float far);
	virtual ~Ortho();

	virtual const Plane GetTopPlane(const Camera& r_camera) const;
    virtual const Plane GetBottomPlane(const Camera& r_camera) const;
    virtual const Plane GetLeftPlane(const Camera& r_camera) const;
    virtual const Plane GetRightPlane(const Camera& r_camera) const;

    virtual void Update(int width, int height);
private:
    void Update();
private:
    float mLeft;
    float mRight;
    float mBottom;
    float mTop;
};

}
#endif /* PROJECTION_ORTHO_H_ */
