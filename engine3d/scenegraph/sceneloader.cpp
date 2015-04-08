#include <math.h>
#include <typeinfo>
#include <list>

#include "sceneloader.h"
#include "engine3d/meshes/triangles.h"
#include "engine3d/meshes/trianglestrip.h"
#include "engine3d/meshes/trianglefan.h"
#include "engine3d/meshes/vertex.h"
#include "engine3d/containment/boundingpolygon.h"
#include "engine3d/containment/boundingsphere.h"
#include "utils/utils.h"

using namespace engine3d;

SceneLoader::SceneLoader(Scene& rScene,
		const String& rProgramFileName,
		const String& rSceneFileName,
		const String& rObjFileName) :
    mrScene(rScene),
	mProgramParser(rProgramFileName),
    mSceneParser(rSceneFileName),
    mObjParser(rObjFileName),
    mShapes("Predefined shapes"),
	mpProgram(nullptr)
{
}

void SceneLoader::Load() 
{
    Debug("Loading objects...");
	LoadProgram(*(mProgramParser.Load()));
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
    Debug("SceneLoader::Load 1");
    if (rElement.mType == TRIANGLE) {
        Mesh* p_mesh = dynamic_cast<Mesh*>(pObject);
        assert(p_mesh);
    	LoadTriangle(rElement, p_mesh);
    } else if (rElement.mType == COORDS) {
    	LoadCoords(rElement, pObject);
    } else if (rElement.mType == TEXCOORDS) {
        Vertex* p_vertex = dynamic_cast<Vertex*>(pObject);
        assert(p_vertex);
        LoadTexCoords(rElement, p_vertex);
    } else if (rElement.mType == USETEXTURE) {
        Shape* p_shape = dynamic_cast<Shape*>(pObject);
        assert(p_shape);
        LoadUseTexture(rElement, p_shape);
    } else if (rElement.mType == USESHAPE) {
        Node* p_node = dynamic_cast<Node*>(pObject);
        assert(p_node);
    	LoadUseShape(rElement, p_node);
    } else if (rElement.mType == ROTATION) {
        Spatial* p_spatial = dynamic_cast<Spatial*>(pObject);
        assert(p_spatial);
        LoadRotation(rElement, p_spatial);
    } else if (rElement.mType == TRANSLATION) {
        Spatial* p_spatial = dynamic_cast<Spatial*>(pObject);
        assert(p_spatial);
        LoadTranslation(rElement, p_spatial);
    } else if (rElement.mType == TEXTURE) {
        Debug(String("Creating new texutre: ") + rElement.mName + ", " + rElement.mParams.GetProperty("file"));
        Texture* p_texture = new Texture(rElement.mParams.GetProperty("file"));
        //if (rElement.mParams.GetProperty("alphafile").Length() > 0) {
        //    Texture tex_alpha(rElement.mParams.GetProperty("alphafile"), rElement.mName + "_alpha", Texture::FORMAT_ALPHA);
        //    Debug("before modulate");
        //    //p_texture->Modulate(tex_alpha);
        //    Debug("after modulate");
        //}
        mTextures.insert({rElement.mName, p_texture});
    } else if (rElement.mType == VERTEX) {
    	LoadVertex(rElement, pObject);
    } else {
        Object* p_object = nullptr;
		if (rElement.mType == NODE) {
			Debug(String("Creating new node: ") + rElement.mName);
			p_object = new Node(rElement.mName);
		} else if (rElement.mType == TRIANGLES) {
			Debug(String("Creating new triangle mesh"));
			p_object = new Triangles();
		} else if (rElement.mType == TRIANGLESTRIP) {
			Debug(String("Creating new trianglestrip"));
			p_object = new TriangleStrip();
		} else if (rElement.mType == TRIANGLEFAN) {
			Debug(String("Creating new trianglefan"));
			p_object = new TriangleFan();
		} else if (rElement.mType == SHAPE) {
			Debug(String("Creating new shape: ") + rElement.mName);
			p_object = new Shape(rElement.mName);
		} else if (rElement.mType == BOUND) {
			Node* p_parent_node = dynamic_cast<Node*>(pObject);
			assert(p_parent_node);
			p_object = LoadBound(rElement, p_parent_node);
		} else {
			Debug(String("Unknown element type: ") + rElement.mType);
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
    }
    Debug("SceneLoader::Load 24");
}

void SceneLoader::LoadProgram(XmlParser::XmlElement& rElement) {
	list<XmlParser::XmlElement*> r_sub_elements = rElement.GetSubElements();
	assert(r_sub_elements.size() == 2);
	string vertex_shader;
	string fragment_shader;
	for (auto i = r_sub_elements.begin(); i != r_sub_elements.end(); i++) {
		if ((*i)->mType == VERTEXSHADER) {
			vertex_shader = Utils::ReadTextFile((*i)->GetParameter("script"));
		} else if ((*i)->mType == FRAGMENTSHADER) {
			fragment_shader = Utils::ReadTextFile((*i)->GetParameter("script"));
		}
	}
	assert(vertex_shader.length() > 0);
	assert(fragment_shader.length() > 0);
	mpProgram = new Program(vertex_shader, fragment_shader);
	assert(mpProgram);
}

void SceneLoader::LoadTriangle(XmlParser::XmlElement& rElement, Mesh* pMesh) {
    Debug("SceneLoader::Load 2");
    Debug(String("indices: ") + rElement.mData);
    ArrayList<String> data = rElement.mData.Split(",");
    Debug("SceneLoader::Load 3");
    int indices[] = {data[0].ToInt(), data[1].ToInt(), data[2].ToInt()};
    Debug(String("Triangle index1: ") + (double)indices[0]);
    Debug(String("Triangle index2: ") + (double)indices[1]);
    Debug(String("Triangle index3: ") + (double)indices[2]);
    pMesh->AddIndices(indices, 3);
    Debug("SceneLoader::Load 4");
    Debug("SceneLoader::Load 5");
}

void SceneLoader::LoadCoords(XmlParser::XmlElement& rElement, Object* pObject) {
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

void SceneLoader::LoadTexCoords(XmlParser::XmlElement& rElement, Vertex* pVertex) {
    Debug("SceneLoader::Load 13");
    ArrayList<String> data = rElement.mData.Split(",");
    Debug("SceneLoader::Load 14");
    double s = data[0].ToDouble();
    double t = data[1].ToDouble();
    Debug(String("Vertex s: ") + s);
    Debug(String("Vertex t: ") + t);
    pVertex->SetTexCoords(s, t);
    Debug("SceneLoader::Load 15");
    Debug("SceneLoader::Load 16");
}

void SceneLoader::LoadUseTexture(XmlParser::XmlElement& rElement, Shape* pShape) {
    Debug(String("Setting texture: ") + rElement.mName);
    Texture* p_tex = GetTexture(rElement.mName);
    assert(p_tex);
    pShape->SetTexture(p_tex);
}

void SceneLoader::LoadUseShape(XmlParser::XmlElement& rElement, Node* pNode) {
    Debug(String("Using shape: ") + rElement.mName);
    Shape* p_shape = dynamic_cast<Shape*>(mShapes.GetChild(rElement.mName));
    assert(p_shape);
    pNode->AddChild(p_shape->Clone());
}

void SceneLoader::LoadRotation(XmlParser::XmlElement& rElement, Spatial* pSpatial) {
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
    Debug(pSpatial->Name());
    Transformation t = pSpatial->GetTransformation();
    t.SetRotation(Vector(x, y, z), a * M_PI / 180);
    pSpatial->SetTransformation(t);
}

void SceneLoader::LoadTranslation(XmlParser::XmlElement& rElement, Spatial* pSpatial) {
    Debug(String("Translation: ") + rElement.mData);
    ArrayList<String> data = rElement.mData.Split(",");
    double x = data[0].ToDouble();
    double y = data[1].ToDouble();
    double z = data[2].ToDouble();
    Debug(String("Translation x: ") + x);
    Debug(String("Translation y: ") + y);
    Debug(String("Translation z: ") + z);
    Debug(pSpatial->Name());
    Transformation t = pSpatial->GetTransformation();
    t.SetTranslation(Vector(x, y, z));
    pSpatial->SetTransformation(t);
}

Vector SceneLoader::LoadVector(XmlParser::XmlElement& rElement) {
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
    return Vector(x, y, z);
}

void SceneLoader::LoadVertex(XmlParser::XmlElement& rElement, Object* pObject) {
	Vector v = LoadVector(rElement);
    if (typeid(*pObject) == typeid(BoundingPolygon)) {
        Debug(String("vertex for boundingpolygon"));
        BoundingPolygon* p_bound = dynamic_cast<BoundingPolygon*>(pObject);
        Debug("SceneLoader::Load 22");
        p_bound->AddVertex(v);
        Debug("SceneLoader::Load 23");
    }
    else if (typeid(*pObject) == typeid(Mesh)) {
        Debug(String("vertex for mesh"));
        Mesh* p_mesh = dynamic_cast<Mesh*>(pObject);
        Debug("SceneLoader::Load 22");
        p_mesh->AddVertex(v);
    }
}

BoundingPolygon* SceneLoader::LoadBound(XmlParser::XmlElement& rElement, Node* pNode) {
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
    //------------------------------------------
    if (usages.Contains("culling")) {
        Object::Dbg("culling bound");
        pNode->SetBound(p_bound);
    }
    if (usages.Contains("collision")) {
        Object::Dbg("collision bound");
        pNode->SetCollisionBound(p_bound);
    }
	return p_bound;
}

Texture* SceneLoader::GetTexture(const string& rName)
{
	auto i = mTextures.find(rName);
	if (i != mTextures.end()) {
		return i->second;
	}
    return nullptr;
}
