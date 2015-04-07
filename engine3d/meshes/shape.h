// Class automatically generated by Dev-C++ New Class wizard

#ifndef SHAPE_H
#define SHAPE_H

#include "base/list.h"
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
		// class constructor
		Shape(const String& name = "");
		Shape(const Shape& s);
		// class destructor
		virtual ~Shape();
	protected:
		void Copy(const Shape& shape);
	public:
		virtual Shape* Clone() const;
		// No description
		//void SetTexCoords(vector<Vector*>* coords);
		virtual void Render();
		// No description
		void SetTexture(Texture* pTexture);
		Texture* GetTexture();
		void SetColor(const Color& rColor);
		void RemoveColor();
		const Color* GetColor() const {return mpColor;};
	protected:
        void BeginRender();  
        void EndRender();
	protected:
        Texture* mpTexture;
        Color* mpColor;
        //List<Vector*> texCoords;
};
}
#endif // SHAPE_H
