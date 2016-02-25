#ifndef VALIS_TEST_SCREEN_H
#define VALIS_TEST_SCREEN_H

#include "Screen.cuh"
#include "GLLibraries.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "SDFRenderer.cuh"

class PBO;
class Texture2D;
class CudaGLBufferMapping;
class dim3;
class SDFRenderer;
class ShaderProgram;
class Camera;
class Player;
class SignedDistanceFunction;
class SDFExtractor;
class VAO;
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
	SDFRenderer* renderer;
	PBO *pbo;
	Texture2D *texture;
	CudaGLBufferMapping *mapping;
	dim3 *windowBlockSize;
	dim3 *windowGridSize;
	ShaderProgram* shader;
	Player* player;
	SignedDistanceFunction* sphereSdf;
	SDFExtractor* extractor;
	VAO* vao;
	VBO* vbo;
	int pointCount;
};

#endif