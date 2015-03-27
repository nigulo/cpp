// Class automatically generated by Dev-C++ New Class wizard

#include "transformationinterpolator.h" // class's header file
#include "engine3d/scenegraph/scene.h" // class's header file

using namespace engine3d;

// class constructor
TransformationInterpolator::TransformationInterpolator(const Scene& rScene, double startTime, double endTime, double period) :
    TransformationController(rScene),
    mStartTime(startTime),
    mEndTime(endTime),
    mPeriod(period),
    mpTrajectory(0)
{
	// insert your code here
}

// class destructor
TransformationInterpolator::~TransformationInterpolator()
{
	// insert your code here
}

void TransformationInterpolator::SetTrajectory(ParametricCurve* pTrajectory)
{
    mpTrajectory = pTrajectory;
}

void TransformationInterpolator::Execute() 
{
    if (!mpTrajectory) {
        return;
    }
    double time = mrScene.GetTime();
    if (time < mStartTime) {
        return;
    }
    if (mEndTime >= mStartTime && time >= mEndTime) {
        // time is over, no movement
        return;
    }
    while (time >= mStartTime + mPeriod) {
        time -= mPeriod;
    }
    double parameter = (time - mStartTime) / mPeriod;
    Vector translation = mpTrajectory->GetPoint(parameter);
    Translate(translation);
}