#include "TestRelativeScreen.cuh"

#include <map>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "GLLibraries.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ShaderProgram.cuh"
#include "Camera.cuh"
#include "Player.cuh"

#include "Descriptor.cuh"
#include "SDFHilbertExtractor.cuh"
#include "CompactMortonPoint.cuh"
#include "IBO.cuh"

#include "SDFHost.cuh"
#include "SDFDevice.cuh"
#include "PlaceSDPrimitive.cuh"
#include "SDSphere.cuh"
#include "SDCube.cuh"

#include "BufferedObjectUsage.cuh"

#include "Texture1D.cuh"

void
TestRelativeScreen::onCreate()
{

	// Load vertex shader
	ifstream myfile("MortonCompactedCloudRenderShader.vert");
	std::stringstream buffer;
	buffer << myfile.rdbuf();
	string vertShader = buffer.str();

	// Load fragment shader
	ifstream myfile2("MortonCompactedCloudRenderShader.frag");
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
	SDCube sdCube(glm::vec3(0.1f, 0.1f, 0.1f), glm::vec3(1, 1, 1), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0, 1, 0), 0);
	//SDTorus sdTorus2(0.33f, 0.07f, glm::vec3(1, 1, 1), glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(0, 1, 0), 0);
	place = new PlaceSDPrimitive();
	testSDF = new SDFHost(&sdCube, 30);
	testSDF->modify(&sdTorus, place);
	testSDFDevice = testSDF->copyToDevice();

	// Create the extractor
	extractor = new SDFHilbertExtractor(128, 128);

	ibo = new IBO(1000000, BufferedObjectUsage::DYNAMIC_DRAW);
	pbo = new PBO(100000);
	pboMapping = new CudaGLBufferMapping<uint32_t>(*pbo, cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
	mapping = new CudaGLBufferMapping<CompactMortonPoint>(*ibo, cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);

	pboTexture = new Texture1D(10000);


	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); // Not a texture. default is modulate.
	
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

	testOffset += 0.01f;
	SDTorus sdTorus2(0.33f, 0.07f, glm::vec3(1, 1, 1), glm::vec3(testOffset, 0.5f, 0.5f), glm::vec3(0, 1, 0), 0);
	testSDF->modify(&sdTorus2, place);
	testSDFDevice = testSDF->copyToDevice();

	testSDF->popEdit();

	pointCount = extractor->extract(*(player->deviceEditSDF), *mapping, *pboMapping, 8);

	
	glm::mat4 invViewProjection;
	player->camera->constructInverseViewProjection(invViewProjection);

	glm::mat4 viewProjection;
	player->camera->constructViewProjection(viewProjection);

	glActiveTexture(GL_TEXTURE0);
	pbo->bind();
	pboTexture->bind();
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Don't sample the border color.
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexSubImage1D(GL_TEXTURE_1D, 0, 0, (GLsizei)(pointCount + 63) / 64, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
	//glTexImage1D(GL_TEXTURE_1D, 0, GL_R32UI, (GLsizei)(pointCount + 63) / 64, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, 0);
	//glBindTexture(GL_TEXTURE_1D, 0);

	//pbo->unbind();

	shader->bind();


	GLint offsetTextureLocation = shader->getUniformLocation("offsetTexture");
	shader->setUniform1i(offsetTextureLocation, 0);

	GLint projectionLocation = shader->getUniformLocation("projectionMatrix");
	shader->setUnifromMatrix4f(projectionLocation, viewProjection);

	GLint resolutionLocation = shader->getUniformLocation("gridResolution");
	shader->setUniformf(resolutionLocation, 128);

	GLint offstSizeLocation = shader->getUniformLocation("offsetBufferSize");
	shader->setUniformf(offstSizeLocation, (pointCount + 63) / 64);

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