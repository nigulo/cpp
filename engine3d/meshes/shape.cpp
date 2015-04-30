#include "shape.h"
#include <GL/gl.h>

using namespace engine3d;

Shape::Shape(const string& name) : Node(name),
	    mpTexture(nullptr),
	    mpColor(nullptr)
{

}

Shape::Shape(const Shape& rShape) : Node(rShape) {
	mpTexture = rShape.mpTexture;
	mpColor = rShape.mpColor;
}

void Shape::Copy(const Shape& rShape)
{
    Node::Copy(rShape);
    mpTexture = rShape.mpTexture;
    mpColor = rShape.mpColor;
}

Shape* Shape::Clone() const
{
    Debug("Shape.Clone");
    Shape* p_shape = new Shape();
    p_shape->Copy(*this);
    return p_shape;
}

Shape::~Shape()
{
}

void Shape::Render()
{
    if (mpTexture) {
    	mpTexture->Use();
    }
    else if (mpColor) {
		glColor4d(mpColor->red, mpColor->green, mpColor->blue, mpColor->alpha);
    }
    Node::Render();
}

// No description
void Shape::SetTexture(Texture* pTexture)
{
    mpTexture = pTexture;
}

Texture* Shape::GetTexture()
{
    return mpTexture;
}

void Shape::SetColor(const Color& rColor)
{
    if (mpColor) {
        delete mpColor;
    }
    mpColor = new Color(rColor.red, rColor.green, rColor.blue, rColor.alpha);
}

void Shape::RemoveColor()
{
    if (mpColor) {
        delete mpColor;
    }
    mpColor = 0;
}
