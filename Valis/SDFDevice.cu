/*
#include "SDFDevice.cuh"

SDFDevice::SDFDevice(DistancePrimitive** primitives, SDModification** modifications, size_t modificationCount) : primitives(primitives), modifications(modifications), modificationCount(modificationCount)
{

}

float
SDFDevice::distanceFromPoint(glm::vec3 position)
{
	float distance = primitives[0]->distanceFromPoint(position);

	for (int i = 0; i < modificationCount; ++i)
	{
		float distance2 = primitives[i + 1]->distanceFromPoint(position); //COME BACK AND FIGURE OUT HOW TO INDEX PROPPERLY
		distance = modifications[i]->modify(distance, distance2); //COME BACK AND FIGURE OUT HOW TO INDEX PROPPERLY
	}

	return distance;
}
*/