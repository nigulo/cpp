#ifndef SHAPE_H
#define SHAPE_H

#include "engine3d/scenegraph/node.h"
#include "engine3d/program/texture.h"
#include "engine3d/attributes/color.h"

using namespace base;

namespace engine3d {

/*
 * Base class for geometric shapes (e.g. Sphere).
 * Contains information about texture associated 
 * with this shape.
 */
class Shape : public Node
{
	public:
		Shape(const string& name = "");
		Shape(const Shape& s);
		virtual ~Shape();
	protected:
		void Copy(const Shape& shape);
	public:
		virtual Shape* Clone() const;
		virtual void Update() {Node::Update();}
		virtual void Render();
		// No description
		void SetTexture(Texture* pTexture);
		Texture* GetTexture();
		void SetColor(const Color& rColor);
		void RemoveColor();
		const Color* GetColor() const {return mpColor;};
	protected:
        Texture* mpTexture;
        Color* mpColor;
};
}
#endif // SHAPE_H
