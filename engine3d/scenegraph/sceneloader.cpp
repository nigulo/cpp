#include <math.h>
#include <typeinfo>

#include "sceneloader.h"
#include "engine3d/meshes/triangles.h"
#include "engine3d/meshes/trianglestrip.h"
#include "engine3d/meshes/trianglefan.h"
#include "engine3d/meshes/vertex.h"
#include "engine3d/containment/boundingpolygon.h"
#include "engine3d/containment/boundingsphere.h"

using namespace engine3d;

SceneLoader::SceneLoader(Scene& rScene, const String& rSceneFileName, const String& rObjFileName) : 
    mrScene(rScene),
    mObjParser(rObjFileName),
    mSceneParser(rSceneFileName),
    mShapes("Predefined shapes")
{
}

void SceneLoader::Load() 
{
    Debug("Loading objects...");
    for (int i = 0; i  < mObjParser.Load()->mSubElements.Size(); i++) {
    	Load(*(mObjParser.Load()->mSubElements[i]), &mShapes);
    }
    Debug("Objects loaded.");
    Debug("Loading scene...");
	Load(*(mSceneParser.Load()->mSubElements[0]));
    Debug("Scene loaded.");
    mTextures.clear();
}

void SceneLoader::Load(XmlParser::XmlElement& rElement, Object* pObject)
{
    Debug("SceneLoader::Load 0");
    Object* p_object = nullptr;
    void* p_void = nullptr; // all objects that don't derive from base::Object

    Debug("SceneLoader::Load 1");
    if (rElement.mType == TRIANGLE) {
        Debug("SceneLoader::Load 2");
        Debug(String("indices: ") + rElement.mData);
        ArrayList<String> data = rElement.mData.Split(",");
        Debug("SceneLoader::Load 3");
        int indices[] = {data[0].ToInt(), data[1].ToInt(), data[2].ToInt()};
        Debug(String("Triangle index1: ") + (double)indices[0]);
        Debug(String("Triangle index2: ") + (double)indices[1]);
        Debug(String("Triangle index3: ") + (double)indices[2]);
        Mesh* p_mesh = dynamic_cast<Mesh*>(pObject);
        assert(p_mesh);
        p_mesh->AddIndices(indices, 3);
        Debug("SceneLoader::Load 4");
        //delete[] p_data;
        Debug("SceneLoader::Load 5");
    }
    else if (rElement.mType == COORDS) {
        Debug("SceneLoader::Load 9");
        ArrayList<String> data = rElement.mData.Split(",");
        Debug("SceneLoader::Load 10");
        double x = data[0].ToDouble();
        double y = data[1].ToDouble();
        double z = data[2].ToDouble();
        Debug(String("Vertex x: ") + x);
        Debug(String("Vertex y: ") + y);
        Debug(String("Vertex z: ") + z);
        if (typeid(*pObject) == typeid(Vertex)) {
            Vertex* p_vertex = dynamic_cast<Vertex*>(pObject);
            assert(p_vertex);
            p_vertex->SetCoords(x, y, z);
        }
        else if (typeid(*pObject) == typeid(Vector)) {
            Vector* p_vector = dynamic_cast<Vector*>(pObject);
            assert(p_vector);
            p_vector->SetCoord(0, x);
            p_vector->SetCoord(1, y);
            p_vector->SetCoord(2, z);
        }
        Debug("SceneLoader::Load 11");
        //delete[] p_data;
        Debug("SceneLoader::Load 12");
    }
    else if (rElement.mType == TEXCOORDS) {
        Debug("SceneLoader::Load 13");
        ArrayList<String> data = rElement.mData.Split(",");
        Debug("SceneLoader::Load 14");
        double s = data[0].ToDouble();
        double t = data[1].ToDouble();
        Debug(String("Vertex s: ") + s);
        Debug(String("Vertex t: ") + t);
        Vertex* p_vertex = dynamic_cast<Vertex*>(pObject);
        assert(p_vertex);
        p_vertex->SetTexCoords(s, t);
        Debug("SceneLoader::Load 15");
        //delete[] p_data;
        Debug("SceneLoader::Load 16");
    }
    else if (rElement.mType == USETEXTURE) {
        Debug(String("Setting texture: ") + rElement.mName);
        Texture* p_tex = GetTexture(rElement.mName);
        assert(p_tex);
        Shape* p_shape = dynamic_cast<Shape*>(pObject);
        assert(p_shape);
        p_shape->SetTexture(p_tex);
    }
    else if (rElement.mType == USESHAPE) {
        //assert(mShapes.GetChild(element.mName));
        Debug(String("Using shape: ") + rElement.mName);
        Shape* p_shape = dynamic_cast<Shape*>(mShapes.GetChild(rElement.mName));
        assert(p_shape);
        p_object = p_shape->Clone();
    }
    else if (rElement.mType == ROTATION) {
        Debug(String("Rotation: ") + rElement.mData);
        ArrayList<String> data = rElement.mData.Split(",");
        double x = data[0].ToDouble();
        double y = data[1].ToDouble();
        double z = data[2].ToDouble();
        double a = data[3].ToDouble();
        Debug(String("RotVector x: ") + x);
        Debug(String("RotVector y: ") + y);
        Debug(String("RotVector z: ") + z);
        Debug(String("RotVector a: ") + a);
        Debug(pObject->Name());
        Spatial* p_spatial = dynamic_cast<Spatial*>(pObject);
        assert(p_spatial);
        Transformation t = p_spatial->GetTransformation();
        t.SetRotation(Vector(x, y, z), a * M_PI / 180);
        p_spatial->SetTransformation(t);
        //delete[] p_data;
    }
    else if (rElement.mType == TRANSLATION) {
        Debug(String("Translation: ") + rElement.mData);
        ArrayList<String> data = rElement.mData.Split(",");
        double x = data[0].ToDouble();
        double y = data[1].ToDouble();
        double z = data[2].ToDouble();
        Debug(String("Translation x: ") + x);
        Debug(String("Translation y: ") + y);
        Debug(String("Translation z: ") + z);
        Debug(pObject->Name());
        Spatial* p_spatial = dynamic_cast<Spatial*>(pObject);
        assert(p_spatial);
        Transformation t = p_spatial->GetTransformation();
        t.SetTranslation(Vector(x, y, z));
        p_spatial->SetTransformation(t);
        //delete[] p_data;
    }
    else if (rElement.mType == TEXTURE) {
        Debug(String("Creating new texutre: ") + rElement.mName + ", " + rElement.mParams.GetProperty("file"));
        Texture* p_texture = new Texture(rElement.mParams.GetProperty("file"));
        //if (rElement.mParams.GetProperty("alphafile").Length() > 0) {
        //    Texture tex_alpha(rElement.mParams.GetProperty("alphafile"), rElement.mName + "_alpha", Texture::FORMAT_ALPHA);
        //    Debug("before modulate");
        //    //p_texture->Modulate(tex_alpha);
        //    Debug("after modulate");
        //}
        mTextures.insert({rElement.mName, p_texture});
    }
    else if (rElement.mType == NODE) {
        Debug(String("Creating new node: ") + rElement.mName);
        p_object = new Node(rElement.mName);
    }
    else if (rElement.mType == VERTEX) {
        Debug(String("Creating new vertex"));
        Object::Dbg(String("Finding property x from : ") + rElement.mParams);
        String sx = rElement.mParams.GetProperty("x");
        Object::Dbg(String("Finding property y from : ") + rElement.mParams);
        String sy = rElement.mParams.GetProperty("y");
        Object::Dbg(String("Finding property z from : ") + rElement.mParams);
        String sz = rElement.mParams.GetProperty("z");
        double x = 0;
        double y = 0;
        double z = 0;
        if (sx.Length() > 0) {
            x = sx.ToDouble();
        }
        if (sy.Length() > 0) {
            y = sy.ToDouble();
        }
        if (sz.Length() > 0) {
            z = sz.ToDouble();
        }
        if (typeid(*pObject) == typeid(BoundingPolygon)) {
            Debug(String("vertex for boundingpolygon"));
            p_void = new Vector(x, y, z);
        }
        else {
            Debug(String("vertex for node"));
            p_object = new Vertex(x, y, z);
        }
    }
    else if (rElement.mType == TRIANGLES) {
        Debug(String("Creating new triangle mesh"));
        p_object = new Triangles();
    }
    else if (rElement.mType == TRIANGLESTRIP) {
        Debug(String("Creating new trianglestrip"));
        p_object = new TriangleStrip();
    }
    else if (rElement.mType == TRIANGLEFAN) {
        Debug(String("Creating new trianglefan"));
        p_object = new TriangleFan();
    }
    else if (rElement.mType == SHAPE) {
        Debug(String("Creating new shape: ") + rElement.mName);
        p_object = new Shape(rElement.mName);
    }
    else if (rElement.mType == BOUND) {
        String type = rElement.mParams.GetProperty("type");
        ArrayList<String> usages = rElement.mParams.GetProperty("usage").Split(",");
        Debug(String("Creating new bound: ") + rElement.mName + "; " + type + "; " + rElement.mParams.GetProperty("usage"));
        BoundingVolume* p_bound;
        if (type.Length() <= 0 || type == "polygon") {
            p_bound = new BoundingPolygon();
        }
        else if (type == "sphere") {
            double x = rElement.mParams.GetProperty("x").ToDouble();
            double y = rElement.mParams.GetProperty("y").ToDouble();
            double z = rElement.mParams.GetProperty("z").ToDouble();
            double r = rElement.mParams.GetProperty("r").ToDouble();
            p_bound = new BoundingSphere(Vector(x, y, z), r);
        }
        Debug(String("Creating bound for node"));
        p_object = p_bound;
        //------------------------------------------
        Node* p_parent_node = dynamic_cast<Node*>(pObject);
        assert(p_parent_node);
        if (usages.Contains("culling")) {
            Object::Dbg("culling bound");
            p_parent_node->SetBound(p_bound);
        }
        if (usages.Contains("collision")) {
            Object::Dbg("collision bound");
            p_parent_node->SetCollisionBound(p_bound);
        }
    }
    else {
        Debug(String("Unknown element type: ") + rElement.mType);
    }
    
    Debug("SceneLoader::Load 17");
    String compile = rElement.mParams.GetProperty("compile");
    if (compile == "yes" || compile == "true" || compile == "1") {
        Node* p_node = dynamic_cast<Node*>(p_object);
        assert(p_node);
        p_node->Compile();
    }
    Debug("SceneLoader::Load 18");
    Node* p_node = dynamic_cast<Node*>(p_object);
    if (p_node) {
        Debug("SceneLoader::Load 19");
        Node* p_parent_node = dynamic_cast<Node*>(pObject);
        if (!p_parent_node) {
            Debug(String("First node"));
            mrScene.SetNode(p_node);
        }
        else {
            Debug(String("Not first node"));
            p_parent_node->AddChild(p_node);
        }
        Debug("SceneLoader::Load 20");
    }
    if (p_object) {
        Debug("SceneLoader::Load 20.1");
    	for (int i = 0; i < rElement.mSubElements.Size(); i++) {
            Load(*rElement.mSubElements[i], p_object);
        }
        Debug("SceneLoader::Load 21");
    }
    if (pObject && p_object) {
        if (typeid(*pObject) == typeid(BoundingPolygon)) {
            BoundingPolygon* p_bound = dynamic_cast<BoundingPolygon*>(pObject);
            Debug("SceneLoader::Load 22");
            Vector* p_vector = static_cast<Vector*>(p_void);
            p_bound->AddVertex(*p_vector);
            delete p_vector;
            Debug("SceneLoader::Load 23");
        }
    }
    Debug("SceneLoader::Load 24");
}

Texture* SceneLoader::GetTexture(const string& rName)
{
	auto i = mTextures.find(rName);
	if (i != mTextures.end()) {
		return i->second;
	}
    return nullptr;
}
