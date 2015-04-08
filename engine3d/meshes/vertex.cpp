#include "vertex.h"

using namespace engine3d;

Vertex::Vertex() : mCoords(0, 0, 0)
{
    mpColor = nullptr;
    mpTexCoords = nullptr;
}

Vertex::Vertex(float x, float y, float z) : mCoords(x, y, z)
{
    mpColor = nullptr;
    mpTexCoords = nullptr;
}

Vertex::Vertex(const Vector& v) : mCoords(v) {
    mpColor = nullptr;
    mpTexCoords = nullptr;
}

Vertex::Vertex(const Vertex& v) : mCoords(v.mCoords) {
    mpColor = nullptr;
    mpTexCoords = nullptr;
    if (v.mpTexCoords) {
        mpTexCoords = new Vector(*v.mpTexCoords);
    }
    if (v.mpColor) {
        mpColor = new Color(v.mpColor->red, v.mpColor->green, v.mpColor->blue, v.mpColor->alpha);
    }
}

Vertex::Vertex(const Vertex& vertex, const Vector& vect) : mCoords(vect) {
    mpColor = nullptr;
    mpTexCoords = nullptr;
    if (vertex.mpTexCoords) {
        mpTexCoords = new Vector(*vertex.mpTexCoords);
    }
    if (vertex.mpColor) {
        mpColor = new Color(vertex.mpColor->red, vertex.mpColor->green, vertex.mpColor->blue, vertex.mpColor->alpha);
    }
}

Vertex* Vertex::Clone() const
{
    return new Vertex(*this);
}

// class destructor
Vertex::~Vertex()
{
    if (mpTexCoords) {
        delete mpTexCoords;
    }
    if (mpColor) {
        delete mpColor;
    }
}

void Vertex::SetColor(const Color& c)
{
    if (mpColor) {
        delete mpColor;
    }
    mpColor = new Color(c.red, c.green, c.blue, c.alpha);
}

void Vertex::SetCoords(const Vector& coords)
{
    assert(coords.GetDim() == 3);
    this->mCoords = coords;
}

void Vertex::SetCoords(float x, float y, float z)
{
    this->mCoords = Vector(x, y, z);
}

void Vertex::SetTexCoords(const Vector& texCoords)
{
    assert(texCoords.GetDim() == 2);
    SetTexCoords(texCoords[0], texCoords[1]);
}

void Vertex::SetTexCoords(float s, float t)
{
    if (mpTexCoords) {
        delete mpTexCoords;
    }
    if (s < 0.0f) {
        s = 0.0f;
    }
    if (t < 0.0f) {
        t = 0.0f;
    }
    if (s > 1.0f) {
        s = 1.0f;
    }
    if (t > 1.0f) {
        t = 1.0f;
    }
    this->mpTexCoords = new Vector(s, t);
}
