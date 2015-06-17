#include "uniform.h"
#include "program.h"
#include "base/object.h"

using namespace engine3d;

Uniform::Uniform(Program& rProgram, const std::string& rName) {
	mId = glGetUniformLocation(rProgram.GetId(), rName.data());
	base::Object::Dbg(to_string(mId) + " = glGetUniformLocation(" + to_string(rProgram.GetId()) + ", " + rName + ")");
}

Uniform::~Uniform() {

}
