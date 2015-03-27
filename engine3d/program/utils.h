#ifndef PROGRAM_UTILS_H_
#define PROGRAM_UTILS_H_

#include <GL/glew.h>
#include <string>

namespace engine3d {

class Utils {
public:
	static const std::string InfoLog(GLuint object,
		    PFNGLGETSHADERIVPROC glGet__iv,
		    PFNGLGETSHADERINFOLOGPROC glGet__InfoLog);
};

} /* namespace engine3d */

#endif /* PROGRAM_UTILS_H_ */
