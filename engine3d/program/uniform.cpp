#include "uniform.h"
#include "program.h"

using namespace engine3d;

Uniform::Uniform(Program& rProgram, const std::string& rName) {
	mId = glGetUniformLocation(rProgram.GetId(), rName.data());
}

Uniform::~Uniform() {

}
