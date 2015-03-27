// Class automatically generated by Dev-C++ New Class wizard

#ifndef NODE_H
#define NODE_H

#include "engine3d/geometry/spatial.h"
#include "engine3d/geometry/transformation.h"
//#include "scene.h"
#include "engine3d/geometry/vector.h"
#include "engine3d/containment/boundingvolume.h"

using namespace base;

namespace engine3d {

class Scene;

/*
 * No description
 */
class Node : public Spatial
{
	public:
		// class constructor
		Node();
		Node(const String& name);
		virtual Node* Clone() const;
		// class destructor
		virtual ~Node();
		// No description
		//void Rotate(double angle, const Vector& axis);
		/**
		 * Sets the parent for the given node. 
		 * Root node does not have a parent
		 */
		//void SetParent(Node* parent);
		/**
		 * Called by Scene::SetNode
		 * Do not call this method directly
		 */
		void SetScene(Scene* parent);
		
		/**
		 * Initializes the node. This method is called 
         * before the Render. It calculates world 
         * transformation and coordinates for this node 
         * and children
		 **/
		virtual void Init();
		virtual void Render();
		/**
		 * Specifies that this node must be compiled into a display list.
		 * This node is compiled into a display list in the Init() method 
         * if it's not already done. listCode parameter is defined there.
		 **/
		void Compile();
		Scene& GetScene() const;
		Node* GetParent() const;
		void AddChild(Node* n);
		void RemoveChild(Node* pNode);
		void RemoveChild(int i);
		Node& GetChild(int i) const;
		Node* GetChild(const String& name) const;
		LinkedList<Node*>& GetChildren() {
			return mChildren;
		}
		// Adds a new child index
		void AddIndex(int index);
		// Adds a set of child indices
		void AddIndices(const int* indices, int count);
		
		bool IsChanged();
		bool IsLeaf() const;
        
        void CheckCollisions();
        
        void SetBound(BoundingVolume* pBound) {
            mpBound = pBound;
        }
        
        void SetCollisionBound(BoundingVolume* pCollisionBound) {
            mpCollisionBound = pCollisionBound;
        }
        
	protected:
        void Copy(const Node& node);
        static bool CheckCollisions(Node& rNode1, Node& rNode2);

        // gets current world transformation for given node
        Transformation GetWorldTransformation() const;
        // gets new world transformation for given node
        Transformation GetNewWorldTransformation() const;

	protected:
        /**
         * Pointer to the parent node. Must be NULL, if
         * mpScene is not NULL and vice versa.
         **/
        Node* mpParent;
        /**
         * Pointer to the scene object. Must be NULL, if
         * mpParent is not NULL and vice versa.
         **/
        Scene* mpScene;
        /**
         * Child nodes of this node
         **/
        LinkedList<Node*> mChildren;
        /**
         * Child indices that edetermine the rendering sequence
         **/
        LinkedList<int> mIndices;

        /**
         * Bounding volume of this node. Bounding volume is
         * used for object culling before rendering.
         **/
        BoundingVolume* mpBound;

        BoundingVolume* mpCollisionBound;

        /**
         * Specifies whether this node can be 
         * OpenGL compiled before rendering
         **/
        bool mCompile;
        /**
         * Code of the OpenGL display list or -1 if this
         * node is not precompiled
         **/
        int mListCode;
        
};
}
#endif // NODE_H
