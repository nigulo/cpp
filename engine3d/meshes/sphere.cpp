#include "sphere.h"
#include "engine3d/scenegraph/camera.h"
#include "engine3d/scenegraph/scene.h"

#include <math.h>
#include <cassert>

using namespace engine3d;

Sphere::Sphere(float radius, int parts1, int parts2, bool genTexCoords, float completeness) :
    rings(parts1 - 2)
{
    assert(radius >= 0);
    assert(parts1 >= 10);
    assert(parts2 >= 10);
    assert(completeness >= 0 && completeness <= 1);
    top = new Triangles();
    bottom = new Triangles();
	this->radius = radius;
	this->parts1 = parts1;
	this->parts2 = parts2;
	this->completeness = completeness;
	
    float dTheta = M_PI / parts1;
    float theta = dTheta;
    float dPhi = 2.0f * M_PI / parts2;
    float phi = 0.0f;
    float ds = 1.0f / parts2;
    float dt = 1.0f / parts1;
    float s = 0.0f;
    float t = 1.0f - dt / 2.0f;

    // Create top
    float y = radius * cos(theta);
    float r = sqrt(radius * radius - y * y);
	for (int i = 0; i < parts2; i++) {
        float z1 = r * cos(phi);
        float x1 = r * sin(phi);
        float z2 = r * cos(phi + dPhi);
        float x2 = r * sin(phi + dPhi);
        if (i == 0) {
            z1 = r;
            x1 = 0.0f;
        }
        else if (i == parts2 - 1) {
            z2 = r;
            x2 = 0.0f;
        }
        Vertex v1(0, radius, 0);
	    if (genTexCoords) {
            v1.SetTexCoords(Vector(s + ds / 2, 1));
        }
        Vertex v2(x1, y, z1);
	    if (genTexCoords) {
            v2.SetTexCoords(Vector(s, t));
        }

        Vertex v3(x2, y, z2);
	    if (genTexCoords) {
            v3.SetTexCoords(Vector(s + ds, t));
        }
        top->Add(v1, v2, v3);
        phi += dPhi;
        s += ds;
    }
    
	float yStop = radius - (2 * radius * completeness);
    // Create middle rings
    for (int i = 0; i < parts1 - 2 && y >= yStop; i++) {
        TriangleStrip* ring = new TriangleStrip();
        float r1 = sqrt(radius * radius - y * y);
        float y2 = radius * cos(theta + dTheta);
        float r2 = sqrt(radius * radius - y2 * y2);
        phi = 0;
        s = 0;
        for (int j = 0; j < parts2 + 1; j++) {
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);
            float z1 = r1 * cosPhi;
            float x1 = r1 * sinPhi;
            float z2 = r2 * cosPhi;
            float x2 = r2 * sinPhi;
            if (j == 0) {
                z1 = r1;
                x1 = 0.0f;
            }
            else if (j == parts2) {
                z2 = r2;
                x2 = 0.0f;
            }
            Vertex v1(x1, y, z1);
            if (genTexCoords) {
                v1.SetTexCoords(Vector(s, t));
            }
            ring->AddVertex(v1);
            Vertex v2(x2, y2, z2);
            if (genTexCoords) {
                v2.SetTexCoords(Vector(s, t - dt));
            }
            ring->AddVertex(v2);
            phi += dPhi;
            s += ds;
        }
        rings.push_back(ring);
        theta += dTheta;
        y = radius * cos(theta);
        t -= dt;
    }
	
    if (y >= yStop) {
	    // Create bottom
	    phi = 0.0f;
	    s = 1.0f;
	    t = dt / 2.0f;
	    r = sqrt(radius * radius - y * y);
		for (int i = 0; i < parts1; i++) {
	        float z1 = r * cos(phi);
	        float x1 = r * sin(phi);
	        float z2 = r * cos(phi - dPhi);
	        float x2 = r * sin(phi - dPhi);
	        if (i == 0) {
	            z1 = r;
	            x1 = 0.0f;
	        }
	        else if (i == parts1 - 1) {
	            z2 = r;
	            x2 = 0.0f;
	        }
	        Vertex v1(0, -radius, 0);
		    if (genTexCoords) {
	            v1.SetTexCoords(Vector(s - ds / 2, 0));
	        }
	        Vertex v2(x1, y, z1);
		    if (genTexCoords) {
	            v2.SetTexCoords(Vector(s, t));
	        }
	
	        Vertex v3(x2, y, z2);
		    if (genTexCoords) {
	            v3.SetTexCoords(Vector(s - ds, t));
	        }
	        bottom->Add(v1, v2, v3);
	        phi -= dPhi;
	        s -= ds;
	    }
    }
	//-------------------------------------------
	// Attach the sphere parts to the sphere node
	//AddChild(&top);
    //for (int i = 0; i < rings.size(); i++) {
    //    AddChild(rings[i]);
    //}
    //AddChild(&bottom);
}

void Sphere::Copy(const Sphere& sphere)
{
    Shape::Copy(sphere);
	this->radius = sphere.radius;
	this->parts1 = sphere.parts1;
	this->parts2 = sphere.parts2;
	this->completeness = sphere.completeness;
	this->top = sphere.top->Clone();
    for (int i = 0; i < sphere.rings.size(); i++) {
        rings.push_back(sphere.rings[i]->Clone());
    }
	this->bottom = sphere.bottom->Clone();
}

Sphere* Sphere::Clone() const
{
    Debug("Sphere.Clone");
    Sphere* p_sphere = new Sphere(radius, parts1, parts2);
    p_sphere->Copy(*this);
    return p_sphere;
}

Sphere::~Sphere()
{
    delete top;
	for (int i = 0; i < rings.size(); i++) {
        delete rings[i];
    }
    rings.clear();
    delete bottom;
}

void Sphere::Render() 
{
    Debug("Sphere::Draw");
    const Color* p_color = GetColor();
    RemoveColor();
    Shape::Render();
    if (p_color) {
    	top->SetColor(*p_color);
    }
    top->Render();
    for (int i = 0; i < rings.size(); i++) {
        if (p_color) {
        	rings[i]->SetColor(*p_color);
        }
        rings[i]->Render();
    }
    if (p_color) {
    	bottom->SetColor(*p_color);
    }
    bottom->Render();
    if (p_color) {
    	SetColor(*p_color);
    }
}
