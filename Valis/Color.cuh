#ifndef VALIS_COLOR_H
#define VALIS_COLOR_H

#include <algorithm>

#include "device_launch_parameters.h"

class Color
{
public:
	float r, g, b, a;

	__host__ __device__
	Color() : r(0), g(0), b(0), a(1)
	{

	}
	__host__ __device__
	Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a)
	{

	}

	__device__ unsigned int
	device_toInt()
	{
		r = clamp(r);   // clamp to [0.0, 1.0]
		g = clamp(g);
		b = clamp(b);
		a = clamp(a);

		return (unsigned int(a * 255) << 24) | (unsigned int(b * 255) << 16) | (unsigned int(g * 255) << 8) | unsigned int(r * 255);
	}

private:
	__host__ __device__ inline float
	clamp(float f)
	{
		return fminf(fmaxf(f, 0.0f), 1.0f);
	}

};


#endif //VALIS_COLOR_H