#include "Galaxy.cuh"

#include "IBO.cuh"
#include "VBO.cuh"
#include "PBO.cuh"
#include "ShaderProgram.cuh"

using namespace glm;

Galaxy::Galaxy(uint32_t resolution)
{

}

Galaxy::~Galaxy()
{

}

void
Galaxy::render(ShaderProgram& shader, glm::mat4& viewProjection)
{
	GLint offsetTextureLocation = shader.getUniformLocation("offsetTexture");
	shader.setUniform1i(offsetTextureLocation, 0);

	GLint projectionLocation = shader.getUniformLocation("projectionMatrix");
	shader.setUnifromMatrix4f(projectionLocation, viewProjection);

	GLint resolutionLocation = shader.getUniformLocation("gridResolution");
	shader.setUniformf(resolutionLocation, 128);

	GLint offstSizeLocation = shader.getUniformLocation("offsetBufferSize");
	shader.setUniformf(offstSizeLocation, (starCount + 63) / 64);

	GLint compactDataAttribute = shader.getAttributeLocation("in_CompactData");

	indices->bind();
	points->bind();

	glEnableVertexAttribArray(compactDataAttribute);
	glEnableClientState(GL_VERTEX_ARRAY);

	glVertexAttribIPointer(compactDataAttribute, 1, GL_UNSIGNED_INT, 0, (void*)(sizeof(uint32_t) * 0));

	glDrawElements(GL_PATCHES, starCount, GL_UNSIGNED_INT, (void*)0);

	glDisableClientState(GL_VERTEX_ARRAY);
	points->unbind();
	indices->unbind();
}