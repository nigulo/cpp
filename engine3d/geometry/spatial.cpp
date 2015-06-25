#include "spatial.h"

using namespace engine3d;

Spatial::Spatial(const string& rName) : Object(rName),
    mChanged(true)
{
}

Spatial::Spatial(const Spatial& rSpatial) 
{
    Copy(rSpatial);
}

void Spatial::Copy(const Spatial& rSpatial)
{
    mChanged = true;
    mOldTransformation = rSpatial.mOldTransformation;
    mTransformation = rSpatial.mTransformation;
    mOldWorldTransformation = rSpatial.mOldWorldTransformation;
    mWorldTransformation = rSpatial.mWorldTransformation;
}

Spatial::~Spatial()
{
}

void Spatial::SetTransformation(const Transformation& rT)
{
	mTransformation = rT;
	mChanged = true;
}

void Spatial::SetWorldTransformation(const Transformation& rT)
{
	mWorldTransformation = rT;
}

const Transformation& Spatial::GetTransformation() const
{
    return mTransformation;
}

const Transformation& Spatial::GetOldTransformation() const
{
    return mOldTransformation;
}

const Transformation& Spatial::GetWorldTransformation() const
{
    return mWorldTransformation;
}

const Transformation& Spatial::GetOldWorldTransformation() const
{
    return mOldWorldTransformation;
}

void Spatial::Transform()
{
    mOldTransformation = mTransformation;
    mOldWorldTransformation = mWorldTransformation;
    mPosition = mTransformation.Transform(mPosition);
}

void Spatial::Revert()
{
    mTransformation = mOldTransformation;
    mWorldTransformation = mOldWorldTransformation;
	mChanged = false;
}

