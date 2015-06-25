#include "body.h"

using namespace engine3d;

Body::Body(Spatial& rSpatial, float mass) :
		mrSpatial(rSpatial), mMass(mass)
{
}

Body::~Body()
{
}

void Body::SetForce(const Vector& rForce)
{
	mForce = rForce;
}

void Body::Move(double dt) {
    mVelocity += mForce / mMass * dt;
    Vector dr = mVelocity * dt;
   	mrSpatial.SetTransformation(Transformation(dr) * mrSpatial.GetTransformation());
}

//void Body::Render() {
//    Move(GetScene().GetTimeChange());
//    Node::Render();
//}

//void Body::CollisionWith(const Node& rNode) {
//	Node::CollisionWith(rNode);
//    const Body* p_body = dynamic_cast<const Body*>(&rNode);
//    if (p_body) {
//
//    }
//}
