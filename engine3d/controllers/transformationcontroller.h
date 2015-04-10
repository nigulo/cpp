#ifndef TRANSFORMATIONCONTROLLER_H
#define TRANSFORMATIONCONTROLLER_H

#include "base/object.h"
#include "engine3d/geometry/vector.h"
#include "engine3d/scenegraph/node.h"
#include "controller.h"
#include <vector>

using namespace base;

namespace engine3d {
/*
 * No description
 */
class TransformationController : public Controller
{
	public:
		/**
		 * @param relative specifies if the transformations are
		 * applied to the current transformations of targets. 
         * Otherwise target transformations are owerloaded.
		 **/
		TransformationController(const Scene& rScene, bool relative = false);
		// class destructor
		~TransformationController();
		/**
		 * Adds target spatial object.
		 **/
		void AddTarget(Spatial* target);
		/**
		 * Rotates the attached targets around the given
		 * axis by a given angle
		 **/
		void Rotate(const Vector& rotVector, double rotAngle);
		/**
		 * Translates the attached targets by a given vector       
		 **/
		void Translate(const Vector& trVector);
		void Transform(const Vector& rotVector, double rotAngle, const Vector& trVector);
	protected:
        vector<Spatial*> mTargets;
        bool mRelative;
};
}
#endif // TRANSFORMATIONCONTROLLER_H
