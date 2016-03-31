#include "Galaxy.cuh"

#include <stdint.h>

#include "IBO.cuh"
#include "PBO.cuh"
#include "ShaderProgram.cuh"

using namespace glm;

Galaxy::Galaxy()
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

	points->bind();
	glEnableVertexAttribArray(compactDataAttribute);
	glEnableClientState(GL_VERTEX_ARRAY);

	glVertexAttribIPointer(compactDataAttribute, 1, GL_UNSIGNED_INT, 0, (void*)(sizeof(uint32_t) * 0));

	glDrawArrays(GL_POINTS, 0, starCount);
	glDisableClientState(GL_VERTEX_ARRAY);
	points->unbind();
}