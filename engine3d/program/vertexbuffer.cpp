/*
 * buffer.cpp
 *
 *  Created on: Mar 27, 2015
 *      Author: nigul
 */

#include "vertexbuffer.h"
#include "engine3d/meshes/vertex.h"
#include "base/object.h"

using namespace engine3d;

VertexBuffer::VertexBuffer(Program& rProgram, int dim, bool textureOrColor) :
		Buffer(GL_ARRAY_BUFFER),
		mDim(dim),
		mrPosition(rProgram.GetAttribute("position")),
		mpTexCoord(nullptr)
{
	if (textureOrColor) {
		mpTexCoord = &rProgram.GetAttribute("texcoord");
	}
}

VertexBuffer::~VertexBuffer() {
}

void VertexBuffer::Render() const {
	base::Object::Dbg("VertexBuffer::Render 1");
    glEnableVertexAttribArray(mrPosition.GetId());
    //glEnableVertexAttribArray(g_resources.flag_program.attributes.normal);
    if (mpTexCoord) {
    	base::Object::Dbg("VertexBuffer::Render 2");
    	glEnableVertexAttribArray(mpTexCoord->GetId());
    }
    //glEnableVertexAttribArray(g_resources.flag_program.attributes.shininess);
    //glEnableVertexAttribArray(g_resources.flag_program.attributes.specular);

    Buffer::Render();
    base::Object::Dbg("VertexBuffer::Render 3");

	GLsizei stride = mDim * sizeof(GLfloat) + (mpTexCoord ? 2 * sizeof(GLfloat) : 0);
    glVertexAttribPointer(
        mrPosition.GetId(),
        3, GL_FLOAT, GL_FALSE, stride,
        (GLvoid*) 0
    );
    //glVertexAttribPointer(
    //    g_resources.flag_program.attributes.normal,
    //    3, GL_FLOAT, GL_FALSE, sizeof(struct flag_vertex),
    //    (void*)offsetof(struct flag_vertex, normal)
    //);
    if (mpTexCoord) {
    	base::Object::Dbg("VertexBuffer::Render 4");
		glVertexAttribPointer(
			mpTexCoord->GetId(),
			2, GL_FLOAT, GL_FALSE, stride,
			(GLvoid*) (mDim * sizeof(GLfloat)) // offset
		);
    }
    //glVertexAttribPointer(
    //    g_resources.flag_program.attributes.shininess,
    //    1, GL_FLOAT, GL_FALSE, sizeof(struct flag_vertex),
    //    (void*)offsetof(struct flag_vertex, shininess)
    //);
    //glVertexAttribPointer(
    //    g_resources.flag_program.attributes.specular,
    //    4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(struct flag_vertex),
    //    (void*)offsetof(struct flag_vertex, specular)
    //);
    //----------------------------------------------
    glDisableVertexAttribArray(mrPosition.GetId());
    //glDisableVertexAttribArray(g_resources.flag_program.attributes.normal);
    if (mpTexCoord) {
    	glDisableVertexAttribArray(mpTexCoord->GetId());
    }
    //glDisableVertexAttribArray(g_resources.flag_program.attributes.shininess);
    //glDisableVertexAttribArray(g_resources.flag_program.attributes.specular);

	}

void VertexBuffer::SetData(const vector<Vertex*>& rVertices) {
	if (rVertices.empty()) {
		return;
	}
	int texDim = mpTexCoord ? 2 : 0;
	GLfloat vertex_coords[rVertices.size()][mDim + texDim];
	//double vertex_tex_coords[mChildren.size()];
	int j = 0;
	for (auto i = rVertices.begin(); i != rVertices.end(); i++) {
        Vertex* p_vertex = *i;
        for (int k = 0; k < mDim; k++) {
        	vertex_coords[j][k] = p_vertex->GetCoord(k);
        	base::Object::Dbg(string("VertexBuffer::SetData coord ") + std::to_string(k) + ": " + std::to_string(vertex_coords[j][k]));
        }
        if (mpTexCoord) {
        	for (int k = 0; k < texDim; k++) {
    			vertex_coords[j][mDim + k] = (*(p_vertex->GetTexCoords()))[k];
            	base::Object::Dbg(string("VertexBuffer::SetData texcoord ") + std::to_string(k) + ": " + std::to_string(vertex_coords[j][mDim + k]));
        	}
        }
	}
	base::Object::Dbg(string("VertexBuffer::SetData ") + std::to_string(rVertices.size() * (mDim + texDim) * sizeof(GLfloat)));
	Buffer::SetData(rVertices.size() * (mDim + texDim) * sizeof(GLfloat), &vertex_coords, GL_STATIC_DRAW);
}
