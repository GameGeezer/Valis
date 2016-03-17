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
	ifstream myfile("PointCloudRenderShader.vert");
	std::stringstream buffer;
	buffer << myfile.rdbuf();
	string vertShader = buffer.str();

	// Load fragment shader
	ifstream myfile2("PointCloudRenderShader.frag");
	std::stringstream buffer2;
	buffer2 << myfile2.rdbuf();
	string fragShader = buffer2.str();

	// Create the shader
	map<int, char *> attributes;
	//attributes.insert(pair<int, char*>(0, "in_CompactData"));
	shader = new ShaderProgram(vertShader.c_str(), fragShader.c_str(), attributes);

	// Create the camera
	Camera* camera = new Camera(640, 680, 0.1f, 100.0f, 45.0f);
	camera->translate(0, 0, 2);

	// Create the player
	player = new Player(*camera);

	// Define an SDF to parse
	SDSphere sdSphere(0.25f, glm::vec3(1, 0.5f, 1), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0, 1, 0), 0);
	SDTorus sdTorus(0.31f, 0.1f, glm::vec3(1, 1, 1), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0, 1, 0), 90);
	//SDTorus sdTorus2(0.33f, 0.07f, glm::vec3(1, 1, 1), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0, 1, 0), 0);
	SDModification* place = new PlaceSDPrimitive();
	SDFHost* testSDF = new SDFHost(&sdSphere, 30);
	testSDF->modify(&sdTorus, place);
	testSDFDevice = testSDF->copyToDevice();

	// Create the extractor
	extractor = new SDFRelativeExtractor(20, 16);

	ibo = new IBO(10000000, BufferedObjectUsage::DYNAMIC_DRAW);
	pbo = new PBO(1000);
	pboMapping = new CudaGLBufferMapping<CompactLocation>(*pbo, cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
	mapping = new CudaGLBufferMapping<CompactRenderPoint>(*ibo, cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
	pointCount = extractor->extract(*testSDFDevice, *mapping, *pboMapping);

	glActiveTexture(GL_TEXTURE0);
	// This code is using the immediate mode texture object 0. Add an own texture object if needed.
	glBindTexture(GL_TEXTURE_2D, 0); // Just use the immediate mode texture.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Don't sample the border color.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); // Not a texture. default is modulate.

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
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->getHandle());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)400, (GLsizei)2, 0, GL_RGBA, GL_UNSIGNED_INT, nullptr);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	
	GLint offsetTextureLocation = shader->getUniformLocation("offsetTexture");
	shader->setUniform1i(offsetTextureLocation, 0);

	GLint projectionLocation = shader->getUniformLocation("projectionMatrix");
	shader->setUnifromMatrix4f(projectionLocation, viewProjection);

	GLint resolutionLocation = shader->getUniformLocation("gridResolution");
	shader->setUniformf(resolutionLocation, 128);

	GLint compactDataAttribute = shader->getAttributeLocation("in_CompactData");
	
	ibo->bind();
	glEnableVertexAttribArray(compactDataAttribute);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	glVertexAttribIPointer(compactDataAttribute, 1, GL_UNSIGNED_INT, 0, (void*)(sizeof(uint32_t) * 0));

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