// Class automatically generated by Dev-C++ New Class wizard

#include "spatial.h" // class's header file

using namespace engine3d;

// class constructor
Spatial::Spatial() :
    mChanged(true) 
{
}

Spatial::Spatial(const String& rName) : 
    mChanged(true), 
    Object(rName)
{
}

Spatial::Spatial(const Spatial& rSpatial) 
{
    Copy(rSpatial);
}

void Spatial::Copy(const Spatial& rSpatial)
{
    mChanged = true;
    mTransformation = rSpatial.mTransformation;
    mWorldTransformation = rSpatial.mWorldTransformation;
    mNewTransformation = rSpatial.mNewTransformation;
    mNewWorldTransformation = rSpatial.mNewWorldTransformation;
}

// class destructor
Spatial::~Spatial()
{
}

void Spatial::SetTransformation(const Transformation& rT)
{
	mNewTransformation = rT;
	//mTransformation = rT;
	mChanged = true;
}

const Transformation& Spatial::GetTransformation() const
{
    return mNewTransformation;
}

const Transformation& Spatial::GetWorldTransformation() const
{
    return mWorldTransformation;
}

//void Spatial::SetNewTransformation(const Transformation& rT)
//{
//	mNewTransformation = rT;
//}
//
//const Transformation& Spatial::GetNewTransformation() const
//{
//    return mNewTransformation;
//}
//
//const Transformation& Spatial::GetNewWorldTransformation() const
//{
//    return mNewWorldTransformation;
//}

void Spatial::Transform()
{
    mTransformation = mNewTransformation;
    mWorldTransformation = mNewWorldTransformation;
}
