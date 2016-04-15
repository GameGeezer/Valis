#ifndef VALIS_GALAXY_CUH
#define VALIS_GALAXY_CUH

#include <glm/mat4x4.hpp>
#include <stdint.h>

class IBO;
class PBO;
class VBO;
class ShaderProgram;

class Galaxy
{
public:
	
	Galaxy(uint32_t resolution);

	~Galaxy();

	void
	render(ShaderProgram& shader, glm::mat4& viewProjection);

private:
	IBO* indices;
	PBO* pointOffsets;
	VBO* points;
	size_t starCount;
};

#endif