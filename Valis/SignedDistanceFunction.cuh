#ifndef VALIS_SIGNED_DISTANCE_FUNCTION_CUH
#define VALIS_SIGNED_DISTANCE_FUNCTION_CUH

#include <glm\vec3.hpp>
#include <glm\vec2.hpp>

#include <vector>

#include "DistancePrimitive.cuh"
#include "GLMUtil.cuh"

class SignedDistanceFunction
{
public:

	SignedDistanceFunction()
	{

	}

	//Only likes one function right now!
	void
	addFunction(DistancePrimitive& primative)
	{
		bounds = primative.calculateBoundingVolume();
		primitives = &primative;
	}

private:
	AABB bounds;
	DistancePrimitive* primitives;
};

#endif