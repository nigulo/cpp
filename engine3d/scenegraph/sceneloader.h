#ifndef SCENELOADER_H
#define SCENELOADER_H

//#include <map>

#include "scene.h"
#include "engine3d/meshes/mesh.h"
#include "engine3d/program/texture.h"
#include "engine3d/containment/boundingvolume.h"
#include "engine3d/projection/projection.h"
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
            void LoadTriangle(XmlParser::XmlElement& rElement, Mesh* pMesh);
            void LoadCoords(XmlParser::XmlElement& rElement, Object* pObject);
            void LoadTexCoords(XmlParser::XmlElement& rElement, Vertex* pVertex);
            void LoadUseTexture(XmlParser::XmlElement& rElement, Shape* pShape);
            void LoadRotation(XmlParser::XmlElement& rElement, Spatial* pSpatial);
            void LoadTranslation(XmlParser::XmlElement& rElement, Spatial* pSpatial);
            Vector LoadVector(XmlParser::XmlElement& rElement);
            Object* LoadVertex(XmlParser::XmlElement& rElement, Object* pObject);
            BoundingVolume* LoadBound(XmlParser::XmlElement& rElement, Node* pNode);
            Shape* LoadUseShape(XmlParser::XmlElement& rElement);
            void LoadProjection(XmlParser::XmlElement& rElement);
            Camera* LoadCamera(XmlParser::XmlElement& rElement);

            Texture* GetTexture(const string& rName);
    	private:
            XmlParser mProgramParser;
            XmlParser mSceneParser;
            XmlParser mObjParser;
            Scene* mpScene;
            map<string, Texture*> mTextures;
            map<string, Projection*> mProjections;
            Node mShapes;
    };
}
#endif
