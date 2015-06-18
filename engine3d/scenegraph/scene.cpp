#include "camera.h"
#include "scene.h"
#include <cassert>
#include <GL/gl.h>

using namespace engine3d;

Scene::Scene() :
	mpProgram(nullptr),
    mpNode(nullptr),
    mpCamera(nullptr)
{
    mPolygonMode[0] = GL_FRONT;
    mPolygonMode[1] = GL_FILL;
    mTime = ((double) GetMillis()) / 1000;
    mTimeChange = 0;
}

Scene::~Scene()
{
}

void Scene::SetNode(Node* pNode) 
{
    pNode->SetScene(this);
    mpNode = pNode;
    if (mpCamera) {
        mpNode->AddChild(mpCamera);
    }
}

void Scene::SetCamera(Camera* pCamera)
{
    mpCamera = pCamera;
    if (mpNode) {
        //mpNode->AddChild(mpCamera);
    }
}

Camera& Scene::GetCamera()
{
    return *mpCamera;
}

const Viewport& Scene::GetViewport() const
{
    return mViewport;
}

void Scene::AddController(Controller* pController) {
    mControllers.push_back(pController);
}

void Scene::Render()
{
    assert(mpNode);
    assert(mpCamera);
    mTimeChange = ((double) GetMillis()) / 1000 - mTime;
    // Execute all controllers
    Debug("Scene::Render gl------------------------------------");
    for (auto&& c : mControllers) {
        c->Execute();
    }
    Debug("Scene::Render 01");
    Debug("Scene::Render 011");
    if (mpNode && (mpNode->IsChanged() || mpCamera->IsChanged())) {
    	if (mpNode->IsChanged()) {
    		mpNode->Update();
    	}
        Debug("Scene::Render 0111");
        mTime += mTimeChange;
        long millis = GetMillis();
        Debug("Scene::Render 1");
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        //glPolygonMode(mPolygonMode[0], mPolygonMode[1]);
        glUseProgram(mpProgram->GetId());
        Debug(string("glUseProgram(") + to_string(mpProgram->GetId()) + ")");

        Debug("Scene::Render 2");
        mpNode->CheckCollisions();
        Debug(string("Node::CheckCollisions took ") + to_string((GetMillis() - millis)));
        Debug("Scene::Render 3");
        mpCamera->Render();
        mpNode->Render();
        Debug(string("Scene::Render took ") + to_string((GetMillis() - millis)));
    }
    Debug("Scene::Render 5");
}

void Scene::SetPolygonMode(int face, int mode)
{
    mPolygonMode[0] = face;
    mPolygonMode[1] = mode;
}
