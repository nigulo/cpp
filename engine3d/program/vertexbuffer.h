/*
 * buffer.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef VERTEXBUFFER_H_
#define VERTEXBUFFER_H_

#include "buffer.h"
#include "program.h"
#include "attribute.h"
#include <vector>

using namespace std;

namespace engine3d {

class Vertex;
class VertexBuffer : public Buffer {
public:
	VertexBuffer(Program& rProgram, int dim = 3, bool textureOrColor = true);
	virtual ~VertexBuffer();

	void SetData(vector<Vertex*> vertices);
	virtual void Render() const;

private:
	int mDim;
	const Attribute& mrPosition;
	const Attribute* mpTexCoord;

};

} /* namespace engine3d */

#endif /* VERTEXBUFFER_H_ */
