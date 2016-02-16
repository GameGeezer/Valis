#ifndef VALIS_KDLEAFNODECOLLECTION_H
#define VALIS_KDLEAFNODECOLLECTION_H

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <class T>
class KDLeafNodeCollection
{
public:
	__host__
	KDLeafNodeCollection();

	__host__ void
	sort();

	__host__ void
	add(uint64_t morton, T& leaf);

	__device__ uint64_t
	getMorton(int32_t index);

	__device__ T*
	getLeaf(int32_t index);

	__host__ __device__ size_t
	size();

private:
	size_t count;
	thrust::device_vector<uint64_t>* device_mortons;
	thrust::device_vector<T*>* device_leaves;

	thrust::host_vector<uint64_t>* host_mortons;
	thrust::host_vector<T*>* host_leaves;
};


#endif //VALIS_KDLEAFNODECOLLECTION_H