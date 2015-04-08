#include "node.h"
#include "scene.h"
#include "camera.h"

#include <GL/gl.h>

using namespace engine3d;

Node::Node() :
    mpParent(nullptr),
    mpScene(nullptr),
    mpBound(nullptr), // no bounding volume by default
    mpCollisionBound(nullptr) // no collision bound by default
{
}

Node::Node(const String& name) :
    Spatial(name),
    mpParent(nullptr),
    mpScene(nullptr),
    mpBound(nullptr), // no bounding volume by default
    mpCollisionBound(nullptr) // no bounding volume by default
{
}

void Node::Copy(const Node& n)
{

    Spatial::Copy(n);
    
    for (auto i = n.mChildren.begin(); i != n.mChildren.end(); i++) {
        AddChild((*i)->Clone());
    }

    if (n.mpBound) {
        mpBound = n.mpBound->Clone();
    }
    if (n.mpCollisionBound) {
        mpCollisionBound = n.mpCollisionBound->Clone();
    }

}

Node* Node::Clone() const
{
    Debug("Node.Clone");
    Node* p_node;// = static_cast<Node*>(o);
    p_node= new Node();
    p_node->Copy(*this);
    return p_node;
}

// class destructor
Node::~Node()
{
	for (int i = 0; i < mChildren.size(); i++) {
        delete mChildren[i];
    }
    mChildren.clear();
}

bool Node::CheckCollisions(Node& rNode1, Node& rNode2) {
    if (!rNode1.mpCollisionBound) {
        return false;
    }
    if (rNode2.mpCollisionBound && rNode1.mpCollisionBound->Collides(*(rNode2.mpCollisionBound))) {
        return true;
    }
    for (auto j = rNode2.mChildren.begin(); j != rNode2.mChildren.end(); j++) {
        if ((*j) == &rNode1) {
            continue;
        }
        if ((*j)->mpCollisionBound && rNode1.mpCollisionBound->Collides(*(*j)->mpCollisionBound)) {
            return true;
        }
        if (CheckCollisions(rNode1, **j)) {
            return true;
        }
    }
    return false;
}


void Node::CheckCollisions()
{
    long millis = GetMillis();
    //Debug("Node::Debug 1");
    if (mpCollisionBound) {
        mpCollisionBound->SetTransformation(GetNewWorldTransformation());
    }
    //Debug("Node::Debug 5");
    for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
        //Debug("Node::Debug 6");
        (*i)->CheckCollisions();
        //Debug("Node::Debug 7");
    }
    for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
        //Debug("Node::Debug 8");
        bool collides = false;
        Node* p_child = *i;
        //Debug("Node::Debug 9");
        for (auto j = mChildren.begin(); j != mChildren.end(); j++) {
            //Debug("Node::Debug 10");
            if ((*j) == p_child) {
                //Debug("Node::Debug 11");
                continue;
            }
            //Debug("Node::Debug 12");
            //if (p_child->mpCollisionBound && (*j)->mpCollisionBound && p_child->mpCollisionBound->Collides(*(*j)->mpCollisionBound)) {
            if (CheckCollisions(*p_child, **j)) {
                //Debug("Node::Debug 13");
                collides = true;
                break;
            }
            //Debug("Node::Debug 14");
        }
        if (!collides) {
            p_child->Transform();
            if (p_child->mpCollisionBound) {
                p_child->mpCollisionBound->Transform();
            }
        }
        else {
        }
    }
    //Debug(String("Node::CheckCollisions took ") + (GetMillis() - millis));
}

void Node::Render()
{
    if (mpBound) {
        mpBound->SetTransformation(GetWorldTransformation());
        mpBound->Transform();
    }
    if (mpBound && GetScene().GetCamera().Cull(*mpBound)) {
        Debug("Object culled");
    }
    else {
		//Debug("transforming");
		mTransformation.Transform();
		// There are no indices defined,
		// render children in the regular order
		for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
			glPushMatrix();
			(*i)->Render();
			glPopMatrix();
		}
    }
    mChanged = false;
    //Debug(String("Node::Render took ") + (GetMillis() - millis));
}

void Node::SetScene(Scene* pScene)
{
    assert(!mpParent);
	this->mpScene = pScene;
}

Node* Node::GetParent() const
{
    return mpParent;
}

Scene& Node::GetScene() const
{
    if (mpParent == NULL) {
        assert(mpScene);
        return *mpScene;
    }
    else {
        return mpParent->GetScene();
    }
}

void Node::AddChild(Node* n) 
{
    assert(n);
    //---------------------
    //char str[17];
    //itoa(mChildren.Size(), str, 10);
    //n->SetName(n->GetName());// + String(str));
    //---------------------
    mChildren.push_back(n);
    n->mpParent = this;
}

Node& Node::GetChild(int i) const
{
    assert(i >= 0 && i < mChildren.size());
    return *mChildren[i];
}

/**
 * @return the child node with the given name or NULL
 * if no such child element exists
 **/
Node* Node::GetChild(const String& name) const
{
    for (int i = 0; i < mChildren.size(); i++) {
        Debug(String("children[") + i + "] = " + mChildren[i]->Name());
        if (mChildren[i]->Name() == name) {
            return mChildren[i];
        }
    }
    return nullptr;
}

void Node::RemoveChild(int i)
{
    assert(i >= 0 && i < mChildren.size());
    assert(mChildren[i]);
    delete mChildren[i];
    mChildren.erase(mChildren.begin() + i);
}

bool Node::IsChanged()
{
    for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
        if ((*i)->IsChanged()) {
            return true;
        }
    }
    return Spatial::IsChanged();
}


bool Node::IsLeaf() const
{
    return mChildren.size() == 0;
}

Transformation Node::GetWorldTransformation() const {
	// TODO:
	return Transformation();
}

Transformation Node::GetNewWorldTransformation() const {
	// TODO:
	return Transformation();
}
