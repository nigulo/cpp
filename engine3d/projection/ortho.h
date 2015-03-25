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
	Ortho(double left, double right, double bottom, double top, double near, double far);
	virtual ~Ortho();

	virtual void Init() const;

	virtual const Plane GetTopPlane(const Camera& r_camera) const;
    virtual const Plane GetBottomPlane(const Camera& r_camera) const;
    virtual const Plane GetLeftPlane(const Camera& r_camera) const;
    virtual const Plane GetRightPlane(const Camera& r_camera) const;

private:
    double mLeft;
    double mRight;
    double mBottom;
    double mTop;
};

}
#endif /* PROJECTION_ORTHO_H_ */
