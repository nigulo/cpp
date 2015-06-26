#include "body.h"

using namespace engine3d;

Body::Body(Node& rNode, float mass, const Vector& rVelocity) :
		mrNode(rNode), mMass(mass), mVelocity(rVelocity)
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
    Debug("Moving body: F" + mForce.ToString() + ", P" + mrNode.GetPosition() + "+" + dr.ToString());
    if (dr.Length2() > 0) {
    	mrNode.SetTransformation(Transformation(dr) * mrNode.GetTransformation());
    }
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
