/*
 * buffer.h
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#ifndef BUFFER_H_
#define BUFFER_H_

#include <GL/glew.h>

namespace engine3d {

class Buffer {
protected:
	Buffer(GLenum type);
	virtual ~Buffer();

public:
	void SetData(GLsizeiptr size, const GLvoid* pData, GLenum usage);

	GLuint GetId() const {
		return mId;
	}

	virtual void Render() const;
protected:
	GLuint mId;
	GLenum mType;

};

} /* namespace engine3d */

#endif /* BUFFER_H_ */
