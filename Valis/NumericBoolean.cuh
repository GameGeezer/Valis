#ifndef NUMERIC_BOOLEAN_CUH
#define NUMERIC_BOOLEAN_CUH

#include <stdint.h>

#include "device_launch_parameters.h"

#define NumericBoolean int32_t

__host__ __device__ __inline__
NumericBoolean numericLessThan_float(float left, float right)
{
	return ((NumericBoolean)(left < right));
}

__host__ __device__ __inline__
NumericBoolean numericGreaterThan_float(float left, float right)
{
	return ((NumericBoolean)(left > right));
}

__host__ __device__ __inline__
NumericBoolean numericIsInRange_int32_t(int32_t value, int32_t lower, int32_t upper)
{
	return ((NumericBoolean)((value <= upper) && value >= lower));
}

__host__ __device__ __inline__
NumericBoolean numericLessThanOrEqual_int32_t(int32_t left, int32_t right)
{
	return ((NumericBoolean)(left <= right));
}

__host__ __device__ __inline__
NumericBoolean numericLessThan_int32_t(int32_t left, int32_t right)
{
	return ((NumericBoolean)(left < right));
}

__host__ __device__ __inline__
NumericBoolean numericGreaterThan_int32_t(int32_t left, int32_t right)
{
	return ((NumericBoolean)(left > right));
}

__host__ __device__ __inline__
NumericBoolean numericGreaterThan_uint32_t(uint32_t left, uint32_t right)
{
	return ((NumericBoolean)(left > right));
}

__host__ __device__ __inline__
NumericBoolean numericGreaterThanOrEqual_uint32_t(uint32_t left, uint32_t right)
{
	return ((NumericBoolean)(left >= right));
}

__host__ __device__ __inline__
NumericBoolean numericLessThan_uint32_t(uint32_t left, uint32_t right)
{
	return ((NumericBoolean)(left < right));
}

__host__ __device__ __inline__
NumericBoolean numericLessThanOrEqual_uint32_t(uint32_t left, uint32_t right)
{
	return ((NumericBoolean)(left <= right));
}

__host__ __device__ __inline__
NumericBoolean numericNotEqual_uint32_t(uint32_t left, uint32_t right)
{
	return ((NumericBoolean)(left != right));
}

__host__ __device__ __inline__
NumericBoolean numericLessThan_uint64_t(uint64_t left, uint64_t right)
{
	return ((NumericBoolean)(left < right));
}

__host__ __device__ __inline__
NumericBoolean numericLessThanOrEqual_uint64_t(uint64_t left, uint64_t right)
{
	return ((NumericBoolean)(left <= right));
}


__host__ __device__ __inline__
NumericBoolean numericGreaterThan_uint64_t(uint64_t left, uint64_t right)
{
	return ((NumericBoolean)(left > right));
}

__host__ __device__ __inline__
NumericBoolean numericNegate_uint32_t(NumericBoolean value)
{
	return (NumericBoolean)(!value);
}

#endif