#ifndef VALIS_BYTE_ARRAY_CUH
#define VALIS_BYTE_ARRAY_CUH

#include <stdint.h>
#include <thrust\device_vector.h>

#include "NumericBoolean.cuh"

#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

struct ByteArrayChunk
{
	uint32_t first, second;
};

class ByteArray
{
public:

	ByteArray(size_t size);

	void
	copyInto(ByteArray& other);

	ByteArrayChunk*
	getDevicePointer();

	void
	zero();

private:

	size_t size;
	ByteArrayChunk* rawDataPointer;
	thrust::device_vector<ByteArrayChunk>* device_data;
};

__device__ __inline__ void
byteArray_setValueAtIndex(ByteArrayChunk* data, uint32_t index, uint32_t value)
{
	//Mask the first byte since that's all we're storing
	value &= 0xF;
	// mod 8, this is a faster alternative that works for powers of 2
	uint32_t byteIndex = index & 7;
	uint32_t writeIndex = index / 8;

	NumericBoolean writeFirst = numericLessThan_uint32_t(byteIndex, 4);
	NumericBoolean writeSecond = numericNegate_uint32_t(writeFirst);

	byteIndex = byteIndex * writeFirst + (byteIndex - 4) * writeSecond;

	uint32_t distanceToShift = 4 * byteIndex;
	uint32_t valueToOr = value << distanceToShift;
	uint32_t andMask = ~(0xF << distanceToShift);

	atomicAnd(&(data[writeIndex].first), (andMask * writeFirst) + (0xFFFF * writeSecond));
	atomicAnd(&(data[writeIndex].second), (andMask * writeSecond) + (0xFFFF * writeFirst));

	atomicOr(&(data[writeIndex].first), valueToOr * writeFirst);
	atomicOr(&(data[writeIndex].second), valueToOr * writeSecond);
}

__device__ __inline__ uint32_t
byteArray_getValueAtIndex(ByteArrayChunk* data, uint32_t index)
{
	// mod 8, this is a faster alternative that works for powers of 2
	uint32_t byteIndex = index & 7;
	uint32_t writeIndex = index / 8;

	NumericBoolean findInFirst = numericLessThan_uint32_t(byteIndex, 4);
	NumericBoolean findInSecond = numericNegate_uint32_t(findInFirst);

	byteIndex = byteIndex * findInFirst + (byteIndex - 4) * findInSecond;

	uint32_t distanceToShift = 4 * byteIndex;
	uint32_t valueToAnd = 0xF << distanceToShift;

	uint32_t firstValue = (data[writeIndex].first & valueToAnd) >> distanceToShift;
	uint32_t secondValue = (data[writeIndex].second & valueToAnd) >> distanceToShift;

	return firstValue * findInFirst + secondValue * findInSecond;
}

#endif