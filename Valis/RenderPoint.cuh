#ifndef VALIS_RENDER_POINT_CUH
#define VALIS_RENDER_POINT_CUH

#include <stdint.h>
#include <glm\vec3.hpp>

struct ExtractionBlock
{
	uint32_t first, second;
};

struct RenderPoint
{
	float positionX, positionY, positionZ;
//	int color;
};

#endif
