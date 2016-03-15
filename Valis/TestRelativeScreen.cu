#include "TestRelativeScreen.cuh"

#include <map>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "GLLibraries.h"

#include "SDFRelativeExtractor.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ShaderProgram.cuh"
#include "Camera.cuh"
#include "Player.cuh"

#include "Descriptor.cuh"
#include "SDFExtractor.cuh"
#include "RenderPoint.cuh"
#include "IBO.cuh"

#include "SDFHost.cuh"
#include "SDFDevice.cuh"
#include "PlaceSDPrimitive.cuh"
#include "SDSphere.cuh"

#include "BufferedObjectUsage.cuh"

void
TestRelativeScreen::onCreate()
{

	// Load vertex shader
	ifstream myfile("BasicShader.vert");
	std::stringstream buffer;
	buffer << myfile.rdbuf();
	string vertShader = buffer.str();

	// Load fragment shader
	ifstream myfile2("BasicShader.frag");
	std::stringstream buffer2;
	buffer2 << myfile2.rdbuf();
	string fragShader = buffer2.str();

	// Create the shader
	map<int, char *> attributes;
	attributes.insert(pair<int, char*>(0, "in_Position"));
	shader = new ShaderProgram(vertShader.c_str(), fragShader.c_str(), attributes);

	// Create the camera
	Camera* camera = new Camera(640, 680, 0.1f, 100.0f, 45.0f);
	camera->translate(0, 0, 2);

	// Create the player
	player = new Player(*camera);

	// Define an SDF to parse
	SDSphere sdSphere(0.25f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDTorus sdTorus(0.31f, 0.1f, glm::vec3(0.5f, 0.5f, 0.5f));
	SDModification* place = new PlaceSDPrimitive();
	SDFHost* testSDF = new SDFHost(&sdSphere);
	testSDF->modify(&sdTorus, place);
	testSDFDevice = testSDF->copyToDevice();

	// Create the extractor
	extractor = new SDFRelativeExtractor(32, 16);

	ibo = new IBO(10000000, BufferedObjectUsage::DYNAMIC_DRAW);
	PBO* pbo = new PBO(1000);
	mapping = new CudaGLBufferMapping<CompactRenderPoint>(*ibo, cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
	pointCount = extractor->extract(*testSDFDevice, *mapping, *pbo);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

void
TestRelativeScreen::onPause()
{

}

void
TestRelativeScreen::onLeave()
{

}

void
TestRelativeScreen::onResume()
{

}

void
TestRelativeScreen::onUpdate(int delta)
{


	player->update(delta);
	glm::mat4 invViewProjection;
	player->camera->constructInverseViewProjection(invViewProjection);

	glm::mat4 viewProjection;
	player->camera->constructViewProjection(viewProjection);

	shader->bind();

	GLint projectionLocation = shader->getUniformLocation("projectionMatrix");
	shader->setUnifromMatrix4f(projectionLocation, viewProjection);

	ibo->bind();
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(1, GL_INT, sizeof(CompactRenderPoint), (void*)(sizeof(uint32_t) * 0));

	glDrawArrays(GL_POINTS, 0, pointCount);
	glDisableClientState(GL_VERTEX_ARRAY);
	ibo->unbind();

	shader->unbind();
}

void
TestRelativeScreen::onResize(int width, int height)
{

}

void
TestRelativeScreen::onDestroy()
{

}