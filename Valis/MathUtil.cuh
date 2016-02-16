#ifndef VALIS_MATH_UTIL_H
#define VALIS_MATH_UTIL_H

class MathUtil
{
public:
	static const float PI;
	static const float PIOver180;
	static const float PIUnder180;

	static inline float
	toDegrees(float radians)
	{
		return radians * PIUnder180;
	}

	static inline float
	toRadians(float degrees)
	{
		return degrees * PIOver180;
	}
};

const float MathUtil::PI = 3.14159265f;

const float MathUtil::PIOver180 = 3.14159265f / 180.0f;

const float MathUtil::PIUnder180 = 180.0f / 3.14159265f;

#endif