#include "sceneloader.h"
#include "engine3d/meshes/triangles.h"
#include "engine3d/meshes/trianglestrip.h"
#include "engine3d/meshes/trianglefan.h"
#include "engine3d/meshes/vertex.h"
#include "engine3d/containment/boundingpolygon.h"
#include "engine3d/containment/boundingsphere.h"
#include "utils/utils.h"

#include <math.h>
#include <typeinfo>
#include <list>
#include <vector>
#include <cassert>


using namespace engine3d;

SceneLoader::SceneLoader(
		const string& rProgramFileName,
		const string& rSceneFileName,
		const string& rObjFileName) :
	mProgramParser(rProgramFileName),
    mSceneParser(rSceneFileName),
    mObjParser(rObjFileName),
    mShapes("Predefined shapes")
{
}

Scene* SceneLoader::Load()
{
	mpScene = new Scene();
    Debug("Loading program...");
	mpScene->SetProgram(LoadProgram(*(mProgramParser.Load())));
    Debug("Loading objects...");
	XmlParser::XmlElement* p_objects = mObjParser.Load();
    for (auto i = p_objects->GetSubElements().begin(); i != p_objects->GetSubElements().end(); i++) {
    	Load(*(*i), &mShapes);
    }
    Debug("Objects loaded.");
    Debug("Loading scene...");
	Load(*(mSceneParser.Load()->GetSubElements().front()));
    Debug("Scene loaded.");
    mTextures.clear();
    return mpScene;
}

void SceneLoader::Load(XmlParser::XmlElement& rElement, Object* pObject)
{
    Debug("SceneLoader::Load 1");
    if (rElement.GetName() == TRIANGLE) {
        Mesh* p_mesh = dynamic_cast<Mesh*>(pObject);
        assert(p_mesh);
    	LoadTriangle(rElement, p_mesh);
    } else if (rElement.GetName() == COORDS) {
    	LoadCoords(rElement, pObject);
    } else if (rElement.GetName() == TEXCOORDS) {
        Vertex* p_vertex = dynamic_cast<Vertex*>(pObject);
        assert(p_vertex);
        LoadTexCoords(rElement, p_vertex);
    } else if (rElement.GetName() == USETEXTURE) {
        Shape* p_shape = dynamic_cast<Shape*>(pObject);
        assert(p_shape);
        LoadUseTexture(rElement, p_shape);
    } else if (rElement.GetName() == USESHAPE) {
        Node* p_node = dynamic_cast<Node*>(pObject);
        assert(p_node);
    	LoadUseShape(rElement, p_node);
    } else if (rElement.GetName() == ROTATION) {
        Spatial* p_spatial = dynamic_cast<Spatial*>(pObject);
        assert(p_spatial);
        LoadRotation(rElement, p_spatial);
    } else if (rElement.GetName() == TRANSLATION) {
        Spatial* p_spatial = dynamic_cast<Spatial*>(pObject);
        assert(p_spatial);
        LoadTranslation(rElement, p_spatial);
    } else if (rElement.GetName() == TEXTURE) {
        Debug(string("Creating new texutre: ") + rElement.GetName() + ", " + *rElement.GetAttribute("file"));
        try {
        	Texture* p_texture = new Texture(mpScene->GetProgram(), rElement.GetName(), *rElement.GetAttribute("file"));
            mTextures.insert({rElement.GetName(), p_texture});
        } catch (const string& ex) {
        	Debug(ex);
        }
        //if (rElement.mParams.GetProperty("alphafile").Length() > 0) {
        //    Texture tex_alpha(rElement.mParams.GetProperty("alphafile"), rElement.GetName() + "_alpha", Texture::FORMAT_ALPHA);
        //    Debug("before modulate");
        //    //p_texture->Modulate(tex_alpha);
        //    Debug("after modulate");
        //}
    } else {
        Object* p_object = nullptr;
        if (rElement.GetName() == VERTEX) {
        	p_object = LoadVertex(rElement, pObject);
        } else if (rElement.GetName() == NODE) {
			Debug(string("Creating new node: ") + rElement.GetName());
			p_object = new Node(rElement.GetName());
		} else if (rElement.GetName() == TRIANGLES) {
			Debug(string("Creating new triangle mesh"));
			p_object = new Triangles();
		} else if (rElement.GetName() == TRIANGLESTRIP) {
			Debug(string("Creating new trianglestrip"));
			p_object = new TriangleStrip();
		} else if (rElement.GetName() == TRIANGLEFAN) {
			Debug(string("Creating new trianglefan"));
			p_object = new TriangleFan();
		} else if (rElement.GetName() == SHAPE) {
			Debug(string("Creating new shape: ") + rElement.GetName());
			p_object = new Shape(rElement.GetName());
		} else if (rElement.GetName() == BOUND) {
			Node* p_parent_node = dynamic_cast<Node*>(pObject);
			assert(p_parent_node);
			p_object = LoadBound(rElement, p_parent_node);
		} else {
			Debug(string("Unknown element type: ") + rElement.GetName());
		}
	    Debug("SceneLoader::Load 18");
	    Node* p_node = dynamic_cast<Node*>(p_object);
	    if (p_node) {
	        Debug("SceneLoader::Load 19");
	        Node* p_parent_node = dynamic_cast<Node*>(pObject);
	        if (!p_parent_node) {
	            Debug(string("First node"));
	            mpScene->SetNode(p_node);
	        }
	        else {
	            Debug(string("Not first node"));
	            p_parent_node->AddChild(p_node);
	        }
	        Debug("SceneLoader::Load 20");
	    }
	    if (p_object) {
	        Debug("SceneLoader::Load 20.1");
	    	for (auto i = rElement.GetSubElements().begin(); i != rElement.GetSubElements().end(); i++) {
	            Load(*(*i), p_object);
	        }
	        Debug("SceneLoader::Load 21");
	    }
    }
    Debug("SceneLoader::Load 24");
}

Program* SceneLoader::LoadProgram(XmlParser::XmlElement& rElement) {
	list<XmlParser::XmlElement*> r_sub_elements = rElement.GetSubElements();
	assert(r_sub_elements.size() == 2);
	string vertex_shader;
	string fragment_shader;
	for (auto&& shader : r_sub_elements) {
		if (shader->GetName() == VERTEXSHADER) {
			vertex_shader = Utils::ReadTextFile(*shader->GetAttribute("script"));
		} else if (shader->GetName() == FRAGMENTSHADER) {
			fragment_shader = Utils::ReadTextFile(*shader->GetAttribute("script"));
		}
	}
	assert(vertex_shader.length() > 0);
	assert(fragment_shader.length() > 0);
	Program* p_program = new Program(vertex_shader, fragment_shader);
	assert(p_program);
	return p_program;
}

void SceneLoader::LoadTriangle(XmlParser::XmlElement& rElement, Mesh* pMesh) {
    Debug("SceneLoader::Load 2");
    Debug(string("indices: ") + rElement.GetInnerText());
    vector<string> data = Utils::Split(rElement.GetInnerText(), ",");
    Debug("SceneLoader::Load 3");
    int indices[] = {stoi(data[0]), stoi(data[1]), stoi(data[2])};
    Debug(string("Triangle index1: ") + to_string(indices[0]));
    Debug(string("Triangle index2: ") + to_string(indices[1]));
    Debug(string("Triangle index3: ") + to_string(indices[2]));
    pMesh->AddIndices(indices, 3);
    Debug("SceneLoader::Load 4");
    Debug("SceneLoader::Load 5");
}

void SceneLoader::LoadCoords(XmlParser::XmlElement& rElement, Object* pObject) {
    Debug("SceneLoader::Load 9");
    vector<string> data = Utils::Split(rElement.GetInnerText(), ",");
    Debug("SceneLoader::Load 10 " + to_string(data.size()));
    double x = stod(data[0]);
    double y = stod(data[1]);
    double z = stod(data[2]);
    Debug(string("Vertex x: ") + to_string(x));
    Debug(string("Vertex y: ") + to_string(y));
    Debug(string("Vertex z: ") + to_string(z));
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
    Debug("SceneLoader::LoadTexCoords 1");
    vector<string> data = Utils::Split(rElement.GetInnerText(), ",");
    double s = stod(data[0]);
    double t = stod(data[1]);
    Debug(string("Vertex s: ") + to_string(s));
    Debug(string("Vertex t: ") + to_string(t));
    pVertex->SetTexCoords(s, t);
    Debug("SceneLoader::LoadTexCoords 2");
}

void SceneLoader::LoadUseTexture(XmlParser::XmlElement& rElement, Shape* pShape) {
    Debug(string("Setting texture: ") + rElement.GetName());
    Texture* p_tex = GetTexture(rElement.GetName());
    assert(p_tex);
    pShape->SetTexture(p_tex);
}

void SceneLoader::LoadUseShape(XmlParser::XmlElement& rElement, Node* pNode) {
    Debug(string("Using shape: ") + rElement.GetName());
    Shape* p_shape = dynamic_cast<Shape*>(mShapes.GetChild(rElement.GetName()));
    assert(p_shape);
    pNode->AddChild(p_shape->Clone());
}

void SceneLoader::LoadRotation(XmlParser::XmlElement& rElement, Spatial* pSpatial) {
    Debug(string("Rotation: ") + rElement.GetInnerText());
    vector<string> data = Utils::Split(rElement.GetInnerText(), ",");
    double x = stod(data[0]);
    double y = stod(data[1]);
    double z = stod(data[2]);
    double a = stod(data[3]);
    Debug(string("RotVector x: ") + to_string(x));
    Debug(string("RotVector y: ") + to_string(y));
    Debug(string("RotVector z: ") + to_string(z));
    Debug(string("RotVector a: ") + to_string(a));
    Debug(pSpatial->Name());
    Transformation t = pSpatial->GetTransformation();
    t.SetRotation(Vector(x, y, z), a * M_PI / 180);
    pSpatial->SetTransformation(t);
}

void SceneLoader::LoadTranslation(XmlParser::XmlElement& rElement, Spatial* pSpatial) {
    Debug(string("Translation: ") + rElement.GetInnerText());
    vector<string> data = Utils::Split(rElement.GetInnerText(), ",");
    double x = stod(data[0]);
    double y = stod(data[1]);
    double z = stod(data[2]);
    Debug(string("Translation x: ") + to_string(x));
    Debug(string("Translation y: ") + to_string(y));
    Debug(string("Translation z: ") + to_string(z));
    Debug(pSpatial->Name());
    Transformation t = pSpatial->GetTransformation();
    t.SetTranslation(Vector(x, y, z));
    pSpatial->SetTransformation(t);
}

Vector SceneLoader::LoadVector(XmlParser::XmlElement& rElement) {
    Debug(string("Creating new vertex"));
    Object::Dbg(string("Finding property x"));
    const string& sx = *rElement.GetAttribute("x");
    Object::Dbg(string("Finding property y"));
    const string& sy = *rElement.GetAttribute("y");
    Object::Dbg(string("Finding property z"));
    const string& sz = *rElement.GetAttribute("z");
    double x = 0;
    double y = 0;
    double z = 0;
    if (sx.length() > 0) {
        x = stod(sx);
    }
    if (sy.length() > 0) {
        y = stod(sy);
    }
    if (sz.length() > 0) {
        z = stod(sz);
    }
    return Vector(x, y, z);
}

Vertex* SceneLoader::LoadVertex(XmlParser::XmlElement& rElement, Object* pObject) {
	Vector v = LoadVector(rElement);
    if (typeid(*pObject) == typeid(BoundingPolygon)) {
        Debug(string("vertex for boundingpolygon"));
        BoundingPolygon* p_bound = dynamic_cast<BoundingPolygon*>(pObject);
        p_bound->AddVertex(v);
    }
    else {
        Mesh* p_mesh = dynamic_cast<Mesh*>(pObject);
        if (p_mesh) {
            Debug(string("vertex for mesh"));
			return &p_mesh->AddVertex(v);
        }
    }
    return nullptr;
}

BoundingVolume* SceneLoader::LoadBound(XmlParser::XmlElement& rElement, Node* pNode) {
    string type = *rElement.GetAttribute("type");
    vector<string> usages = Utils::Split(*rElement.GetAttribute("usage"), ",");
    Debug(string("Creating new bound: ") + rElement.GetName() + "; " + type + "; " + *rElement.GetAttribute("usage"));
    BoundingVolume* p_bound;
    if (type.length() <= 0 || type == "polygon") {
        p_bound = new BoundingPolygon();
    }
    else if (type == "sphere") {
        double x = stod(*rElement.GetAttribute("x"));
        double y = stod(*rElement.GetAttribute("y"));
        double z = stod(*rElement.GetAttribute("z"));
        double r = stod(*rElement.GetAttribute("r"));
        p_bound = new BoundingSphere(Vector(x, y, z), r);
    }
    Debug(string("Creating bound for node"));
    //------------------------------------------
    if (Utils::Contains(usages, string("culling"))) {
        Object::Dbg("culling bound");
        pNode->SetBound(p_bound);
    }
    if (Utils::Contains(usages, string("collision"))) {
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
