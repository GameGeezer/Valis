#include "VAO.cuh"

#include "Descriptor.cuh"
#include "VBO.cuh"

VAO::VAO(VBO& vbo, size_t size) : vbo(&vbo), size(size)
{
	glGenVertexArrays(1, &handle);
}

void
VAO::init()
{
	glBindVertexArray(handle);

	vbo->bind();

	for (auto iter = descriptors.begin(); iter != descriptors.end(); ++iter)
	{
		int index = iter->first;

		Descriptor* descriptor = iter->second;

		glEnableVertexAttribArray(index);

		glVertexAttribPointer(index, descriptor->size, descriptor->type, descriptor->normalized, descriptor->stride, &(descriptor->pointer));
	}

	glBindVertexArray(0);

	vbo->unbind();
}

void
VAO::draw()
{
	vbo->bind();
	glBindVertexArray(handle);
	
	glDrawElements(GL_POINTS, size, GL_UNSIGNED_INT, 0);
	vbo->unbind();
	glBindVertexArray(0);
}

void
VAO::destroy()
{

}

void
VAO::addVertexAttribute(uint32_t index, Descriptor& descriptor)
{
	descriptors.insert(pair<int, Descriptor*>(index, &descriptor));
}