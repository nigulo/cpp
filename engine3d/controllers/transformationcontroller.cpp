#include "transformationcontroller.h"
#include "engine3d/scenegraph/camera.h"
#include <cassert>

using namespace engine3d;

TransformationController::TransformationController(const Scene& rScene, bool relative) :
    Controller(rScene),
    mRelative(relative)
{
}

TransformationController::~TransformationController()
{
}

void TransformationController::AddTarget(Spatial* target)
{
    mTargets.push_back(target);
}

void TransformationController::Rotate(const Vector& rotVector, double rotAngle) 
{
     
    for (auto i = mTargets.begin(); i != mTargets.end(); i++) {
        Spatial* p_target = *i;
        assert(p_target);
        Transformation t1(rotVector, rotAngle);
        if (mRelative) {
            p_target->SetTransformation(p_target->GetTransformation() * t1);
        } else {
            p_target->SetTransformation(t1);
        }
    }
}

void TransformationController::Translate(const Vector& trVector) 
{
    for (auto i = mTargets.begin(); i != mTargets.end(); i++) {
        Spatial* p_target = *i;
        assert(p_target);
        Transformation t1(trVector);
        if (mRelative) {
            p_target->SetTransformation(p_target->GetTransformation() * t1);
        }
        else {
            p_target->SetTransformation(t1);
        }
    }
}

void TransformationController::Transform(const Vector& rotVector, double rotAngle, const Vector& trVector)
{
    for (auto i = mTargets.begin(); i != mTargets.end(); i++) {
        Spatial* p_target = *i;
        assert(p_target);
        Transformation t1(rotVector, rotAngle, trVector);
        if (mRelative) {
            p_target->SetTransformation(p_target->GetTransformation() * t1);
        }
        else {
            p_target->SetTransformation(t1);
        }
    }
}
