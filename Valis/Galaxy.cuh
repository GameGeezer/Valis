#ifndef VALIS_GALAXY_CUH
#define VALIS_GALAXY_CUH

#include <glm/mat4x4.hpp>

class IBO;
class PBO;
class ShaderProgram;

class Galaxy
{
public:
	
	Galaxy();

	~Galaxy();

	void
		render(ShaderProgram& shader, glm::mat4& viewProjection);

private:
	IBO* points;
	PBO* pointOffsets;
	size_t starCount;
};

#endif