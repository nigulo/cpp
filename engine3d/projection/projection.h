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
#include "engine3d/geometry/matrix.h"
#include "engine3d/program/program.h"
#include "engine3d/program/attribute.h"

using namespace base;

namespace engine3d {

class Camera;
class Projection : public Object {

protected:
	Projection(Program& rProgram, float near, float far);

public:
	virtual ~Projection();
	void Project() const;
    /**
     * @return disctance of the near clipping plane measured from eye point
     **/
	float GetZNear() const {
        return mZNear;
    }

    /**
     * @return disctance of the far clipping plane measured from eye point
     **/
	float GetZFar() const {
        return mZFar;
    }

	const Program& GetProgram() const {
		return mrProgram;
	}

	virtual const Plane GetTopPlane(const Camera& r_camera) const = 0;
    virtual const Plane GetBottomPlane(const Camera& r_camera) const = 0;
    virtual const Plane GetLeftPlane(const Camera& r_camera) const = 0;
    virtual const Plane GetRightPlane(const Camera& r_camera) const = 0;

protected:

    /**
     * Disctance of the near clipping plane measured from eye point
     **/
    float mZNear;

    /**
     * Disctance of the far clipping plane measured from eye point
     **/
    float mZFar;

    Matrix mMatrix;

    Program& mrProgram;
    const Attribute& mrAttribute;
};

}
#endif /* PROJECTION_PROJECTION_H_ */
