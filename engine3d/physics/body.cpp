#include "body.h"

using namespace engine3d;

Body::Body(Node& rNode, float mass, const Vector& rVelocity) :
		mrNode(rNode), mMass(mass), mVelocity(rVelocity)
{
	mrNode.AddCollisionListener(*this);
}

Body::~Body()
{
}

void Body::SetForce(const Vector& rForce)
{
	mForce = rForce;
}

void Body::Move(float dt) {
    mVelocity += mForce / mMass * dt;
    Vector dr = mVelocity * dt;
    if (dr.Length2() > 0) {
        Debug("Moving body: F" + mForce.ToString() + ", V" + mVelocity.ToString() + ", P" + mrNode.GetPosition() + "+" + dr.ToString());
    	mrNode.SetTransformation(Transformation(dr) * mrNode.GetTransformation());
    }
}

void Body::Collision(const Node& rNode1, const Node& rNode2, const unique_ptr<Vector>& rPoint) {
	Debug("Body collision: " + rPoint.get()->ToString());

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
