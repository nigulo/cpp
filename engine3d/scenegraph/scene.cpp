#include "camera.h"
#include "scene.h"
#include <GL/gl.h>

using namespace engine3d;

Scene::Scene(Program& rProgram) :
	mrProgram(rProgram),
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
    mControllers.Add(pController);
}

void Scene::Render()
{
    assert(mpNode);
    assert(mpCamera);
    mTimeChange = ((double) GetMillis()) / 1000 - mTime;
    // Execute all controllers
    Debug("Scene::Render 0");
    for (LinkedList<Controller*>::Iterator i = mControllers.Begin(); !i.Done(); i++) {
        (*i)->Execute();
    }
    Debug("Scene::Render 01");
    Debug("Scene::Render 011");
    if (mpNode && (mpNode->IsChanged() || mpCamera->IsChanged())) {
        Debug("Scene::Render 0111");
        mTime += mTimeChange;
        long millis = GetMillis();
        Debug("Scene::Render 1");
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_TEXTURE_2D);
        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPolygonMode(mPolygonMode[0], mPolygonMode[1]);
        glUseProgram(mrProgram.GetId());
        glActiveTexture(GL_TEXTURE0);

        Debug("Scene::Render 2");
        mpNode->CheckCollisions();
        Debug(String("Node::CheckCollisions took ") + (GetMillis() - millis));
        Debug("Scene::Render 3");
        mpCamera->Render();
        mpNode->Render();
        Debug(String("Scene::Render took ") + (GetMillis() - millis));
    }
    Debug("Scene::Render 5");
}

void Scene::SetPolygonMode(int face, int mode)
{
    mPolygonMode[0] = face;
    mPolygonMode[1] = mode;
}
