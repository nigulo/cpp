#ifndef SCENELOADER_H
#define SCENELOADER_H

//#include <map>

#include "scene.h"
#include "engine3d/meshes/shape.h"
#include "engine3d/attributes/texture.h"
#include "utils/xmlparser.h"

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
            void Load(XmlParser::XmlElement& rElement, Object* pObject);
            Texture* GetTexture(const String& name);
    	private:
            XmlParser mSceneParser;
            XmlParser mObjParser;
            Scene& mrScene;
            LinkedList<Texture*> mTextures;
            Node mShapes;
    };
}
#endif
