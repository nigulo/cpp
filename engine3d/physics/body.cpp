// Class automatically generated by Dev-C++ New Class wizard

#include "body.h" // class's header file
#include "engine3d/scenegraph/scene.h" // class's header file

using namespace engine3d;

// class constructor
Body::Body(double mass) : mMass(mass)
{
	// insert your code here
}

// class destructor
Body::~Body()
{
	// insert your code here
}

// No description
void Body::SetForce(const Vector& rForce)
{
	mForce = rForce;
}

void Body::Move(double dt) {
    mVelocity += mForce / mMass * dt;
    Vector dr = mVelocity * dt;
   	mTransformation = Transformation(dr) * mTransformation;
}

void Body::Render() {
    Move(GetScene().GetTimeChange());
    Node::Render();
}

bool Body::IsChanged() {
    if (Node::IsChanged()) {
        return true;
    }
    return GetScene().GetTimeChange() > 0 && mVelocity != Vector();
}
