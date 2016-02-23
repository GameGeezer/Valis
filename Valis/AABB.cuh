#ifndef VALIS_AABB_CUH
#define VALIS_AABB_CUH

#include <glm\vec2.hpp>

class AABB
{
public:
	glm::vec2 lower, upper;

	AABB()
	{

	}

	AABB(glm::vec2 lower, glm::vec2 upper) : lower(lower), upper(upper)
	{

	}
};

#endif