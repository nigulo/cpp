#include "node.h"
#include "scene.h"
#include "camera.h"

#include <cassert>
#include <GL/glew.h>

using namespace engine3d;

Node::Node() :
    mpParent(nullptr),
    mpScene(nullptr),
    mpBound(nullptr), // no bounding volume by default
    mpCollisionBound(nullptr) // no collision bound by default
{
}

Node::Node(const string& name) :
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

Node::~Node()
{
	for (int i = 0; i < mChildren.size(); i++) {
		assert(mChildren[i]);
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
    Debug("Node::CheckCollisions 1");
    if (mpCollisionBound) {
        mpCollisionBound->SetTransformation(GetNewWorldTransformation());
    }
    Debug("Node::CheckCollisions 5");
    for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
        Debug("Node::CheckCollisions 6");
        (*i)->CheckCollisions();
        Debug("Node::CheckCollisions 7");
    }
    for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
        Debug("Node::CheckCollisions 8");
        bool collides = false;
        Node* p_child = *i;
        Debug("Node::CheckCollisions 9");
        for (auto j = mChildren.begin(); j != mChildren.end(); j++) {
            Debug("Node::CheckCollisions 10");
            if ((*j) == p_child) {
                Debug("Node::CheckCollisions 11");
                continue;
            }
            Debug("Node::CheckCollisions 12");
            //if (p_child->mpCollisionBound && (*j)->mpCollisionBound && p_child->mpCollisionBound->Collides(*(*j)->mpCollisionBound)) {
            if (CheckCollisions(*p_child, **j)) {
                Debug("Node::CheckCollisions 13");
                collides = true;
                break;
            }
            Debug("Node::CheckCollisions 14");
        }
        if (!collides) {
            Debug("Node::CheckCollisions 15");
            p_child->Transform();
            if (p_child->mpCollisionBound) {
                p_child->mpCollisionBound->Transform();
            }
        }
        else {
        }
    }
}

void Node::Update() {
	for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
        Debug("Updating child");
		(*i)->Update();
	}
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
		Debug("transforming");
		//mTransformation.Transform();
		//if (mChanged) {
		//	Update();
		//}
		for (auto i = mChildren.begin(); i != mChildren.end(); i++) {
	        Debug("Rendering child");
			(*i)->Render();
		}
    }
    mChanged = false;
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
    if (mpParent == nullptr) {
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
Node* Node::GetChild(const string& name) const
{
    for (int i = 0; i < mChildren.size(); i++) {
        Debug(string("children[") + to_string(i) + "] = " + mChildren[i]->Name());
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
