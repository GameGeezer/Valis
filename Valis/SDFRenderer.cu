#include "SDFRenderer.cuh"

#include "Color.cuh"
#include "CudaGLBufferMapping.cuh"
#include "Camera.cuh"
#include <glm/mat4x4.hpp>
#include "DistanceFunctions.h"
#include "NumericBoolean.cuh"


__global__ void
d_render(int *d_output, int imageW, int imageH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x < imageW) && (y < imageH))
	{
		Color color(0.1f, 0.25f, 1, 1);
		// In our sample tex is always valid, but for something like your own
		// sparse texturing you would need to make sure to handle the zero case.

		// write output color
		int i = y * imageW + x;
		d_output[i] = color.device_toInt();
	}
}

__global__ void
d_RayRender(int *d_output, int imageW, int imageH, glm::mat4 inverseViewProjection)
{
	float x = blockIdx.x * blockDim.x + threadIdx.x;
	float y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = (x / (float)imageW) * 2.0f - 1.0f;
	float v = (y / (float)imageH) * 2.0f - 1.0f;

	if ((x >= imageW) || (y >= imageH))
	{
		return;
	}
	
	Color color(0.3f, 0.3f, 1, 1);
	glm::vec3 origin(u, v, 1);
	glm::vec4 direction = inverseViewProjection * glm::vec4(0, 0, 1, 1);
	NumericBoolean found = 0;
	for (int i = 0; i < 20; ++i)
	{
		//float distance = sdSphere(origin + offset, 0.5f);
		float distance = sdTorous(origin, glm::vec2(0.5f, 0.1f));
		//float distance = sdCylinder(origin + offset, glm::vec3(0.2f, 0.01f, 0.1f));
		//float distance = sdCone(origin + offset, glm::vec2(0.3f, 0.5f));
		glm::vec4 offset = direction * distance;
		origin += glm::vec3(offset.x, offset.y, offset.z);
		found += numericLessThan_float(distance, 0.01f);
	}

	Color color2(0.85f, 0.3f, 0.3f, 1);
	// In our sample tex is always valid, but for something like your own
	// sparse texturing you would need to make sure to handle the zero case.

	// write output color
	int i = y * imageW + x;
	NumericBoolean hitSphere = numericGreaterThan_int32_t(found, 0);
	//d_output[i] = color2.device_toInt();
	d_output[i] = color.device_toInt() * numericNegate_uint32_t(hitSphere) + color2.device_toInt() * hitSphere;
}

void
SDFRenderer::renderToMapping(CudaGLBufferMapping& mapping, dim3 windowGridSize, dim3 windowBlockSize, glm::mat4& inverseViewProjection)
{
	mapping.map();
	d_RayRender << <windowGridSize, windowBlockSize >> >(mapping.getDeviceOutput(), 640, 480, inverseViewProjection);
	mapping.unmap();
}
