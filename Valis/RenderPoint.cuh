#ifndef VALIS_RENDER_POINT_CUH
#define VALIS_RENDER_POINT_CUH

#include <stdint.h>
#include <glm\vec3.hpp>

#include "device_launch_parameters.h"

struct ExtractionBlock
{
	uint32_t first = 0, second = 0;
};

struct RenderPoint
{
	float positionX, positionY, positionZ;
//	int color;
};

#endif
