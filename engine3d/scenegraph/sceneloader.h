#ifndef SCENELOADER_H
#define SCENELOADER_H

//#include <map>

#include "scene.h"
#include "engine3d/meshes/mesh.h"
#include "engine3d/program/texture.h"
#include "engine3d/containment/boundingvolume.h"
#include "utils/xmlparser.h"
#include <map>
#include <string>

#define VERTEXSHADER    "vertexshader"
#define FRAGMENTSHADER  "fragmentshader"
#define NODE            "node"
#define TRIANGLES       "triangles"
#define TRIANGLESTRIP   "trianglestrip"
#define TRIANGLEFAN     "trianglefan"
#define VERTEX          "vertex"
#define COORDS          "coords"
#define TRIANGLE        "triangle"
#define TEXFILE         "texfile"
#define TEXCOORDS       "texcoords"
#define SHAPE           "shape"
#define TEXTURE         "texture"
#define USETEXTURE      "usetexture"
#define USESHAPE        "useshape"
#define ROTATION        "rotation"
#define TRANSLATION     "translation"
#define BOUND           "bound"

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
            SceneLoader(const String& rProgramFileName = "program.xml",
            		const String& rSceneFileName = "scene.xml",
					const String& rObjFileName = "objects.xml");
            Scene* Load();
    	private:
            Program* LoadProgram(XmlParser::XmlElement& rElement);
            void Load(XmlParser::XmlElement& rElement, Object* pObject = nullptr);
            void LoadTriangle(XmlParser::XmlElement& rElement, Mesh* pMesh);
            void LoadCoords(XmlParser::XmlElement& rElement, Object* pObject);
            void LoadTexCoords(XmlParser::XmlElement& rElement, Vertex* pVertex);
            void LoadUseTexture(XmlParser::XmlElement& rElement, Shape* pShape);
            void LoadUseShape(XmlParser::XmlElement& rElement, Node* pNode);
            void LoadRotation(XmlParser::XmlElement& rElement, Spatial* pSpatial);
            void LoadTranslation(XmlParser::XmlElement& rElement, Spatial* pSpatial);
            Vector LoadVector(XmlParser::XmlElement& rElement);
            void LoadVertex(XmlParser::XmlElement& rElement, Object* pObject);
            BoundingVolume* LoadBound(XmlParser::XmlElement& rElement, Node* pNode);

            Texture* GetTexture(const string& rName);
    	private:
            XmlParser mProgramParser;
            XmlParser mSceneParser;
            XmlParser mObjParser;
            Scene* mpScene;
            map<string, Texture*> mTextures;
            Node mShapes;
    };
}
#endif
