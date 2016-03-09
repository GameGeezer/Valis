#ifndef VALIS_TEST_SCREEN_H
#define VALIS_TEST_SCREEN_H

#include "Screen.cuh"
#include "GLLibraries.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "CudaGLBufferMapping.cuh"

class dim3;
class ShaderProgram;
class Camera;
class Player;
class SignedDistanceFunction;
class SDFExtractor;
class VBO;

class TestScreen : public Screen
{
	__host__ void
	onCreate() override;

	__host__ void
	onPause() override;

	__host__ void
	onLeave() override;

	__host__ void
	onResume() override;

	__host__ void
	onUpdate(int delta) override;

	__host__ void
	onResize(int width, int height) override;

	__host__ void
	onDestroy() override;
private:
	GLuint vertexbuffer;
	ShaderProgram* shader;
	Player* player;
	SDFExtractor* extractor;
	VBO* vbo;
	int pointCount;
};

#endif