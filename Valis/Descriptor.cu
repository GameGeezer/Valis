#include "Descriptor.cuh"

Descriptor::Descriptor(int32_t size, int32_t type, bool normalized, int32_t stride, int32_t pointer) : size(size), type(type), normalized(normalized), stride(stride), pointer(pointer)
{

}