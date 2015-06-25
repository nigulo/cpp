#ifndef SCENE_H
#define SCENE_H

#include "base/object.h"
#include "engine3d/program/program.h"
#include "viewport.h"
#include "engine3d/controllers/controller.h"
#include "engine3d/physics/body.h"
#include "engine3d/physics/field.h"
#include <vector>

using namespace std;
using namespace base;

namespace engine3d {

class Node;
class Camera;

class Scene : public Object
{
	public:
		Scene();
		~Scene();
        
        /**
         * @return reference to viewport associated
         * with this scene
         **/
		const Viewport& GetViewport()const;
		

		/**
		 * Sets the root node for this scene. All other
		 * nodes (including camera) are the subnodes of 
         * this root node
         * Scene cannot be rendered
         * before root node is set.
		 **/
		void SetNode(Node* pNode);

		/**
		 * Gets the root node of this scene.
		 **/
		Node* GetNode() const {
            return mpNode;
        }

        /**
         * Sets the camera for this scene. 
         * Scene cannot be rendered
         * before camera is associated with
         * the scene.
         **/
        void SetCamera(Camera* pCamera);

        /**
         * @return reference to camera associated
         * with this scene
         **/
		Camera& GetCamera();

		/**
		 * Renders the scene
		 **/
		void Render();
		
		/**
		 * Sets the polygon mode for this scene.
		 **/
		void SetPolygonMode(int face, int mode);
		
		double GetTimeChange() const {
            return mTimeChange;
        }

		double GetTime() const {
            return mTime;
        }
        
        /**
         * Adds new controller to the scene
         **/
        void AddController(Controller* pController);

        void AddBody(Body* pBody);
        void AddField(Field* pField);
        
        void SetProgram(Program* pProgram) {
        	mpProgram = pProgram;
        }

        Program& GetProgram() {
        	return *mpProgram;
        }

	protected:
        
        Program* mpProgram;
        Camera* mpCamera;
        // root node
        Node* mpNode;
        Viewport mViewport;
        int mPolygonMode[2]; // [0] - face, [1] - mode
        /**
         * Scene object controllers
         **/
        vector<Controller*> mControllers;

        // Physics (later maybe needed to be moved to separate class)
        vector<Body*> mBodies;
        vector<Field*> mFields;

        /**
         * Current time in seconds
         **/
        double mTime;
        
        /**
         * Time elapsed from last rendering in seconds
         **/
        double mTimeChange;
};
}
#endif // SCENE_H
