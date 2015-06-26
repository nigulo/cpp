#ifndef SCENELOADER_H
#define SCENELOADER_H

//#include <map>

#include "scene.h"
#include "engine3d/meshes/mesh.h"
#include "engine3d/program/texture.h"
#include "engine3d/containment/boundingvolume.h"
#include "engine3d/physics/body.h"
//#include "engine3d/physics/field.h"
#include "engine3d/projection/projection.h"
#include "engine3d/controllers/transformationcontroller.h"
#include "utils/xmlparser.h"
#include <map>
#include <string>

using namespace base;
using namespace utils;
using namespace std;

namespace engine3d {

    /**
     * Implements scene loading from XML tree
     */
    class SceneLoader : public Object
    {
    	public:
            // Creates a new SceneLoader using the given scene and object files
            SceneLoader(const string& rProgramFileName = "program.xml",
            		const string& rSceneFileName = "scene.xml",
					const string& rObjFileName = "objects.xml");
            Scene* Load();
    	private:
            Program* LoadProgram(XmlParser::XmlElement& rElement);
            void Load(XmlParser::XmlElement& rElement, Object* pObject = nullptr);
            void LoadTriangle(const XmlParser::XmlElement& rElement, Mesh* pMesh);
            void LoadCoords(const XmlParser::XmlElement& rElement, Object* pObject);
            void LoadTexCoords(const XmlParser::XmlElement& rElement, Vertex* pVertex);
            void LoadUseTexture(const XmlParser::XmlElement& rElement, Shape* pShape);
            void LoadRotation(const XmlParser::XmlElement& rElement, Spatial* pSpatial);
            void LoadTranslation(const XmlParser::XmlElement& rElement, Spatial* pSpatial);
            Vector LoadVector(const XmlParser::XmlElement& rElement);
            Object* LoadVertex(const XmlParser::XmlElement& rElement, Object* pObject);
            BoundingVolume* LoadBound(const XmlParser::XmlElement& rElement, Node* pNode);
            Shape* LoadUseShape(const XmlParser::XmlElement& rElement);
            void LoadProjection(const XmlParser::XmlElement& rElement);
            Camera* LoadCamera(const XmlParser::XmlElement& rElement);
            void LoadController(const XmlParser::XmlElement& rElement);
            void LoadUseController(const XmlParser::XmlElement& rElement, Spatial* pSpatial);
            Body* LoadBody(const XmlParser::XmlElement& rElement, Node* pNode);
            void LoadField(const XmlParser::XmlElement& rElement, Object* pSpatial);

            Texture* GetTexture(const string& rName);
    	private:
            XmlParser mProgramParser;
            XmlParser mSceneParser;
            XmlParser mObjParser;
            Scene* mpScene;
            map<string, Texture*> mTextures;
            map<string, Projection*> mProjections;
            map<string, TransformationController*> mControllers;
            //map<string, Field*> mFields;
            Node mShapes;
    };
}
#endif
