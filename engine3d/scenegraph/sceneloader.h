#ifndef SCENELOADER_H
#define SCENELOADER_H

//#include <map>

#include "scene.h"
#include "engine3d/meshes/shape.h"
#include "engine3d/program/texture.h"
#include "utils/xmlparser.h"
#include <map>
#include <string>

#define SCENE           "scene"
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
            SceneLoader(Scene& rScene, const String& rSceneFileName = "scene.xml", const String& rObjFileName = "objects.xml");
            void Load();
    	private:
            void Load(XmlParser::XmlElement& rElement, Object* pObject = nullptr);
            Texture* GetTexture(const string& rName);
    	private:
            XmlParser mSceneParser;
            XmlParser mObjParser;
            Scene& mrScene;
            map<string, Texture*> mTextures;
            Node mShapes;
    };
}
#endif
