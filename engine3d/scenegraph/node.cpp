// Class automatically generated by Dev-C++ New Class wizard

#include "node.h"
#include "scene.h"
#include "camera.h"

#include <GL/gl.h>

using namespace engine3d;
// class constructor
Node::Node() : 
    mListCode(-1),
    mCompile(false),
    mpParent(0),
    mpScene(0),
    mpBound(0), // no bounding volume by default
    mpCollisionBound(0) // no collision bound by default
{
}

Node::Node(const String& name) : 
    mListCode(-1),
    mCompile(false),
    Spatial(name),
    mpParent(0),
    mpScene(0),
    mpBound(0), // no bounding volume by default
    mpCollisionBound(0) // no bounding volume by default
{
}

void Node::Copy(const Node& n)
{

    Spatial::Copy(n);
    
    for (LinkedList<Node*>::Iterator i = n.mChildren.Begin(); !i.Done(); i++) {
        AddChild((*i)->Clone());
    }

    for (LinkedList<int>::Iterator i = n.mIndices.Begin(); !i.Done(); i++) {
        mIndices.Add(*i);
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
    if (mListCode >= 0) {
        glDeleteLists(mListCode, 1);
    }
	for (int i = 0; i < mChildren.Size(); i++) {
        delete mChildren[i];
    }
    mChildren.Clear();
    mIndices.Clear();
}

bool Node::CheckCollisions(Node& rNode1, Node& rNode2) {
    if (!rNode1.mpCollisionBound) {
        return false;
    }
    if (rNode2.mpCollisionBound && rNode1.mpCollisionBound->Collides(*(rNode2.mpCollisionBound))) {
        return true;
    }
    for (LinkedList<Node*>::Iterator j = rNode2.mChildren.Begin(); !j.Done(); j++) {
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
    for (LinkedList<Node*>::Iterator i = mChildren.Begin(); !i.Done(); i++) {
        //Debug("Node::Debug 6");
        (*i)->CheckCollisions();
        //Debug("Node::Debug 7");
    }
    for (LinkedList<Node*>::Iterator i = mChildren.Begin(); !i.Done(); i++) {
        //Debug("Node::Debug 8");
        bool collides = false;
        Node* p_child = *i;
        //Debug("Node::Debug 9");
        for (LinkedList<Node*>::Iterator j = mChildren.Begin(); !j.Done(); j++) {
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

void Node::Init()
{
    //long millis = GetMillis();
    //Debug("Node::Init 1");
//	if (mpParent) {
//        //Debug("Node::Init 2");
//    	const Transformation parentWT = mpParent->GetWorldTransformation();
//       	mWorldTransformation = Transformation(mTransformation * parentWT);
//    }
//    else {
//        mWorldTransformation = mTransformation;
//        //Debug("Node::Init 4");
//    }
    if (mpBound) {
        mpBound->SetTransformation(GetWorldTransformation());
        mpBound->Transform();
    }
    //Debug("Node::Init 5");
    for (LinkedList<Node*>::Iterator i = mChildren.Begin(); !i.Done(); i++) {
        //Debug("Node::Init 6");
        (*i)->Init();
        //Debug("Node::Init 7");
    }
    //-----------------------------
    // If this node and all subnodes are precompilable
    if (mCompile && mListCode < 0) {
        // This node must be compiled into a display list
        Debug("Compiling...");
        int list_code = glGenLists(1);
        glNewList(list_code, GL_COMPILE);
        Render();
        glEndList();
        mListCode = list_code;
    }
    //Debug(String("Node::Init took ") + (GetMillis() - millis));
}

void Node::Render()
{
    //long millis = GetMillis();
    //Debug("Node::Render");
    if (mListCode >= 0) {
        // This node is a compiled display list
        glCallList(mListCode);
    }
    else if (!mCompile && mpBound && GetScene().GetCamera().Cull(*mpBound)) {
        Debug("Object culled");
    }
    else {
		//Debug("transforming");
		mTransformation.Transform();
        int num_indices = mIndices.Size();
        if (num_indices <= 0) {
            // There are no indices defined,
            // render children in the regular order 
            //int num_children = mChildren.Size();
            //for (int i = 0; i < num_children; i++) {
            for (LinkedList<Node*>::Iterator i = mChildren.Begin(); !i.Done(); i++) {
                glPushMatrix();
                (*i)->Render();
                glPopMatrix();
            }
        }
        else {
            for (LinkedList<int>::Iterator i = mIndices.Begin(); !i.Done(); i++) {
                //Debug(String("index: ") + (*i) + " Node: " + Name());
                assert((*i) < mChildren.Size());
                glPushMatrix();
                //Debug(String("indices[") + i + "]:" + mIndices[i]);
                mChildren[*i]->Render();
                glPopMatrix();
            }
        }
    }
    mChanged = false;
    //Debug(String("Node::Render took ") + (GetMillis() - millis));
}

void Node::Compile()
{
    mCompile = true;
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

//void Node::SetTransformation(const Transformation& t)
//{
//	mTransformation = t;
//	SetChanged();
//}
//
//const Transformation& Node::GetLocalTransformation() const
//{
//    return mTransformation;
//}
//
//const Transformation& Node::GetWorldTransformation() const
//{
//    return mWorldTransformation;
//}

void Node::AddChild(Node* n) 
{
    assert(n);
    //---------------------
    //char str[17];
    //itoa(mChildren.Size(), str, 10);
    //n->SetName(n->GetName());// + String(str));
    //---------------------
    mChildren.Add(n);
    n->mpParent = this;
}

Node& Node::GetChild(int i) const
{
    assert(i >= 0 && i < mChildren.Size());
    return *mChildren[i];
}

/**
 * @return the child node with the given name or NULL
 * if no such child element exists
 **/
Node* Node::GetChild(const String& name) const
{
    for (int i = 0; i < mChildren.Size(); i++) {
        Debug(String("children[") + i + "] = " + mChildren[i]->Name());
        if (mChildren[i]->Name() == name) {
            return mChildren[i];
        }
    }
    return NULL;
}

void Node::RemoveChild(Node* pNode)
{
	//int index = 0;
    //for (LinkedList<Node*>::Iterator i = mChildren.Begin(); !i.Done(); i++) {
    //    if ((*i) == pNode) {
    //    	break
    //    }
    //    index++;
    //}
    //assert(index >= 0 && index < mChildren.Size());
    mChildren.Remove(pNode);
    delete pNode;
}

void Node::RemoveChild(int i)
{
    assert(i >= 0 && i < mChildren.Size());
    assert(mChildren[i]);
    delete mChildren[i];
    mChildren.RemoveAt(i);
}

void Node::AddIndex(int index)
{
    assert(index >= 0);
	mIndices.Add(index);
}

void Node::AddIndices(const int* indices, int count)
{
    for (int i = 0; i < count; i++) {
        assert(indices[i] >= 0);
    	this->mIndices.Add(indices[i]);
    }
}

bool Node::IsChanged()
{
    for (LinkedList<Node*>::Iterator i = mChildren.Begin(); !i.Done(); i++) {
        if ((*i)->IsChanged()) {
            return true;
        }
    }
    return Spatial::IsChanged();
}


bool Node::IsLeaf() const
{
    return mChildren.Size() == 0;
}

Transformation Node::GetWorldTransformation() const {
	// TODO:
	return Transformation();
}

Transformation Node::GetNewWorldTransformation() const {
	// TODO:
	return Transformation();
}