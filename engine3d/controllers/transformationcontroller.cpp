#include "transformationcontroller.h"
#include "engine3d/scenegraph/camera.h"

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
    mTargets.Add(target);
}

void TransformationController::Rotate(const Vector& rotVector, double rotAngle) 
{
     
    for (LinkedList<Spatial*>::Iterator i = mTargets.Begin(); !i.Done(); i++) {
        Spatial* p_target = (*i);
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
    for (LinkedList<Spatial*>::Iterator i = mTargets.Begin(); !i.Done(); i++) {
        Spatial* p_target = (*i);
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
    for (LinkedList<Spatial*>::Iterator i = mTargets.Begin(); !i.Done(); i++) {
        Spatial* p_target = (*i);
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
