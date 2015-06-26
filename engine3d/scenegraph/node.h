#ifndef NODE_H
#define NODE_H

#include "engine3d/geometry/spatial.h"
#include "engine3d/geometry/transformation.h"
#include "engine3d/geometry/vector.h"
#include "engine3d/containment/boundingvolume.h"
#include "engine3d/program/program.h"
#include "engine3d/program/attribute.h"
#include "engine3d/scenegraph/collisionlistener.h"

#include <vector>
#include <memory>

using namespace base;
using namespace std;

namespace engine3d {

class Scene;
class Node : public Spatial
{
	public:
		Node(const string& name = "");
		virtual Node* Clone() const;
		virtual ~Node();
		/**
		 * Called by Scene::SetNode
		 * Do not call this method directly
		 */
		void SetScene(Scene* parent);
		
		// Pre-collision/culling check (updating world transformation)
		virtual void Init();
		// Post-transformation stuff (updating vertex buffers etc.)
		virtual void Update();
		virtual void Render();
		Scene& GetScene() const;
		Node* GetParent() const;
		void AddChild(Node* n);
		void RemoveChild(int i);
		Node& GetChild(int i) const;
		Node* GetChild(const string& name) const;
		vector<Node*>& GetChildren() {
			return mChildren;
		}
		
		bool IsChanged();
		bool IsLeaf() const;
        
        void CheckCollisions();
        virtual void CollisionWith(const Node& rNode, const unique_ptr<Vector>& rPoint);
        
        void SetBound(BoundingVolume* pBound) {
            mpBound = pBound;
        }
        
        void SetCollisionBound(BoundingVolume* pCollisionBound) {
            mpCollisionBound = pCollisionBound;
        }

        void AddCollisionListener(CollisionListener& rCollisionListener) {
        	Debug("AddCollisionListener");
        	mCollisionListeners.push_back(&rCollisionListener);
        }

	protected:
        void Copy(const Node& node);
        static unique_ptr<Vector> CheckCollisions(const Node& rNode1, const Node& rNode2);

        // gets current world transformation for given node
        //Transformation GetWorldTransformation() const;
        // gets new world transformation for given node
        //Transformation GetNewWorldTransformation() const;

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
        vector<Node*> mChildren;

        /**
         * Bounding volume of this node. Bounding volume is
         * used for object culling before rendering.
         **/
        BoundingVolume* mpBound;

        BoundingVolume* mpCollisionBound;
        vector<CollisionListener*> mCollisionListeners;

};
}
#endif // NODE_H
