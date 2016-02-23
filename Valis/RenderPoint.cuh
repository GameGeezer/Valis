#ifndef VALIS_RENDER_POINT_CUH
#define VALIS_RENDER_POINT_CUH

#include <stdint.h>
#include <glm\vec3.hpp>

struct RenderPoint
{
	char positionX, positionY, positionZ;

	__host__ __device__
	inline void setPosition(uint16_t x, uint16_t y, uint16_t z)
	{
		positionX = x;
		positionY = y;
		positionZ = z;
	}

	__host__ __device__
	inline void getPosition(glm::vec3& target)
	{
		target.x = positionX;
		target.y = positionY;
		target.z = positionZ;
	}
};

#endif
