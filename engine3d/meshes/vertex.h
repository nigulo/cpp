#ifndef VERTEX_H
#define VERTEX_H

#include "engine3d/geometry/vector.h"
#include "shape.h"
#include "engine3d/attributes/color.h"

using namespace base;

namespace engine3d {

/*
 * The class containing vertex attributes
 */
class Vertex
{
	public:
		Vertex();
		Vertex(float x, float y, float z);
		Vertex(const Vector& v);
		Vertex(const Vertex& v);
		/**
		 * Creates a vertex using all parameters of other
		 * vertex except for the coordinates which are taken
		 * from the specified vector
		 **/
		Vertex(const Vertex& vertex, const Vector& vect);

		virtual Vertex* Clone() const;
		virtual ~Vertex();
		// Sets vertex color
		void SetColor(const Color& c);
		void SetCoords(const Vector& coords);
		void SetCoords(float x, float y, float z);
		void SetTexCoords(const Vector& texCoords);
		void SetTexCoords(float s, float t);
		const Color* GetColor() const {return mpColor;};
		float GetCoord(int index) const {return (mCoords)[index];}
		const Vector& GetCoords() const {return mCoords;}
		const Vector* GetTexCoords() const {return mpTexCoords;}
	protected:
        Color* mpColor;
        Vector mCoords;
        Vector* mpTexCoords;
};
}
#endif // VERTEX_H
