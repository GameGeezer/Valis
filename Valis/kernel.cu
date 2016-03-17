
#include "cuda_runtime.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "device_launch_parameters.h"

#include "Application.cuh"
#include "Game.cuh"
#include "TestScreen.cuh"
#include "TestRelativeScreen.cuh"

#include "Color.cuh"

#include <stdio.h>

#include <stdint.h>

#include "NumericBoolean.cuh"

#define Morton uint64_t
#define MortonHighestOrder uint64_t
#define Direction int32_t


__host__ __device__ inline
uint64_t highestOrderBit(uint64_t n)
{
	n |= (n >> 1);
	n |= (n >> 2);
	n |= (n >> 4);
	n |= (n >> 8);
	n |= (n >> 16);
	n |= (n >> 32);
	return n - (n >> 1);
}

__host__ __device__ inline
MortonHighestOrder highestOrderBitDifferent(Morton first, Morton second)
{
	return highestOrderBit(first ^ second);
}

__host__ __device__ inline
MortonHighestOrder highestOrderBitSame(Morton first, Morton second)
{
	return highestOrderBit(first & second);
}

__host__ __device__ inline
int32_t closestHighPowerOfTwo(int32_t x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	++x;
	return x;
}

void testDetermineRange3(int32_t index, thrust::host_vector<uint64_t>& device_leafNodes)
{
	++index;

	Morton lowMorton = device_leafNodes[index - 1];
	Morton morton = device_leafNodes[index];
	Morton highMorton = device_leafNodes[index + 1];

	MortonHighestOrder highDifference = highestOrderBitDifferent(morton, highMorton);
	MortonHighestOrder lowDifference = highestOrderBitDifferent(morton, lowMorton);

	// 1 if the high morton is less than the low morton, 0 otherwise
	NumericBoolean isHighLessThanLow = numericLessThan_uint64_t(highDifference, lowDifference);
	// 1 if the low morton is less than the high morton, 0 otherwise
	// NOTE: In no case should the morton codes be equal.
	NumericBoolean isHighGreaterThanLow = numericGreaterThan_uint64_t(highDifference, lowDifference);
	// 1 if to the right, 0 otherwise
	Direction direction = isHighLessThanLow - isHighGreaterThanLow;
	// The highest value between the high and low morton
	MortonHighestOrder greatestMorton = isHighGreaterThanLow * highDifference + isHighLessThanLow * lowDifference;

	int32_t leafCount = device_leafNodes.size() - 1;
	int32_t halfUpPower2 = closestHighPowerOfTwo(device_leafNodes.size()) / 2;

	int32_t startToEndDifference = 0;
	// This shouldn't branch because the number of leaves is consistent
	for (int32_t offset = halfUpPower2; offset >= 1; offset /= 2)
	{
		int32_t newIndex = index + ((startToEndDifference + offset)) * direction;
		// 1 if in the range, 0 otherwise
		NumericBoolean isInRange = numericIsInRange_int32_t(newIndex, 1, leafCount);
		// This more than likely accesses illegal memory. Should I be worried?
		MortonHighestOrder splitPrefix = highestOrderBitDifferent(morton, device_leafNodes[newIndex]);
		NumericBoolean isSplitIsLessThanGreatest = numericLessThan_uint64_t(splitPrefix, greatestMorton);
		// Update the distance between the beginning and the current legal end
		startToEndDifference += offset * isSplitIsLessThanGreatest * isInRange;
	}

	int32_t end = index + startToEndDifference * direction;

	MortonHighestOrder beginToEndMortonDifference = highestOrderBitDifferent(morton, device_leafNodes[end]);
	//Add ine to account for integers rounding down when divided
	++startToEndDifference;
	int32_t cumulativeOffset = 0;
	for (int32_t div = 1; div <= halfUpPower2; div *= 2)
	{
		int32_t offset = (startToEndDifference / div);
		int32_t newIndex = index + (cumulativeOffset + offset) * direction;

		NumericBoolean isInRange = numericIsInRange_int32_t(newIndex, 1, leafCount);

		MortonHighestOrder beginToOffsetMortonDifference = highestOrderBitDifferent(morton, device_leafNodes[newIndex]);

		NumericBoolean offsetGreaterThanEnd = numericLessThan_uint64_t(beginToOffsetMortonDifference, beginToEndMortonDifference);

		cumulativeOffset += (offset * offsetGreaterThanEnd) * isInRange;
	}
	// The split Will be found a "cumulative offset" distance ind "direction" from the index.
	//The nodes moving in a positive direction will find the split one low and the nodes moving in a negative directione
	//Will find the split one high. Adding "isHighLessThanLow" will make all of the nodes round the split high
	cumulativeOffset = (cumulativeOffset * direction) + index + isHighLessThanLow;

	int x = 5;
}

int main()
{
	Screen* screen = new TestScreen();
	Game* game = new Game(*screen);
	Application* application = new Application(*game, "Test!", 640, 480);

	application->start();
	
	uint64_t highDifference = 5;
	uint64_t lowDifference = 10;
	int32_t direction = ((int32_t)((highDifference > lowDifference))) - ((int32_t)((highDifference < lowDifference)));

	thrust::host_vector<uint64_t> device_leafNodes(12);
	device_leafNodes[0] = std::numeric_limits<uint64_t>::max();
	device_leafNodes[1] = 5;
	device_leafNodes[2] = 9;
	device_leafNodes[3] = 13;
	device_leafNodes[4] = 17;
	device_leafNodes[5] = 21;
	device_leafNodes[6] = 24;
	device_leafNodes[7] = 25;
	device_leafNodes[8] = 30;
	device_leafNodes[9] = 32;
	device_leafNodes[10] = 56;
	device_leafNodes[11] = 65;
	//device_leafNodes[12] = 77;

	testDetermineRange3(0, device_leafNodes);
	testDetermineRange3(1, device_leafNodes);
	testDetermineRange3(2, device_leafNodes);
	testDetermineRange3(3, device_leafNodes);
	testDetermineRange3(4, device_leafNodes);
	testDetermineRange3(5, device_leafNodes);
	testDetermineRange3(6, device_leafNodes);
	testDetermineRange3(7, device_leafNodes);
	testDetermineRange3(8, device_leafNodes);
	testDetermineRange3(9, device_leafNodes);
	testDetermineRange3(10, device_leafNodes);
	testDetermineRange3(11, device_leafNodes);
	testDetermineRange3(12, device_leafNodes);

//	cout << "Hello, World!" << endl;
	return 0;
}