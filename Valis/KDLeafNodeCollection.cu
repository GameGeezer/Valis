
#include "KDLeafNodeCollection.cuh"

#include <thrust\sort.h>

template <class T>
KDLeafNodeCollection<T>::KDLeafNodeCollection() : device_mortons(new thrust::device_vector<uint64_t>()), device_leaves(new thrust::device_vector<T*>()),
host_mortons(new thrust::host_vector<uint64_t>()), host_leaves(new thrust::host_vector<T*>())
{
	// device_mortons->push_back(std::numeric_limits<uint64_t>::max());
	// device_leaves->push_back(NULL);
}

template <class T>
void
KDLeafNodeCollection<T>::sort()
{
	*device_mortons = *host_mortons;
	*device_leaves = *host_leaves;
	thrust::sort_by_key(device_mortons->begin(), device_mortons->end(), device_leaves->begin());
}

template <class T>
void
KDLeafNodeCollection<T>::add(uint64_t morton, T& leaf)
{
	++count;
	host_mortons->push_back(morton);
	host_leaves->push_back(&leaf);
}

template <class T>
uint64_t
KDLeafNodeCollection<T>::getMorton(int32_t index)
{
	return device_mortons[index + 1];
}

template <class T>
T*
KDLeafNodeCollection<T>::getLeaf(int32_t index)
{
	return device_leaves[index + 1];
}

template <class T>
size_t
KDLeafNodeCollection<T>::size()
{
	return count;
}