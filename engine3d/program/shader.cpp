#include "shader.h"
#include "utils.h"
#include "base/object.h"
#include <GL/glew.h>

using namespace engine3d;
using namespace std;

Shader::Shader(GLenum type, const string& rShaderScript) {
    const GLchar* source = rShaderScript.data();

    if (rShaderScript.length() == 0) {
		throw ("Empty shader script");
    }

    mId = glCreateShader(type);
    GLint length = rShaderScript.length();
    glShaderSource(mId, 1, (const GLchar**)&source, &length);
    glCompileShader(mId);

    GLint shader_ok;
    glGetShaderiv(mId, GL_COMPILE_STATUS, &shader_ok);
    if (!shader_ok) {
        string info_log = Utils::InfoLog(mId, glGetShaderiv, glGetShaderInfoLog);
        base::Object::Dbg(string("Loading shader failed: ") + info_log);
        glDeleteShader(mId);
		throw (info_log);
    }
}

Shader::~Shader() {
}
