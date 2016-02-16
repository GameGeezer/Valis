#include "Camera.cuh"

#include <math.h>

#include "MathUtil.cuh"
#include <glm/gtx/transform.hpp>

Camera::Camera(float width, float height, float near, float far, float fieldOfView) : width(width), height(height), near(near), far(far), fieldOfView(fieldOfView)
{
	updateProjection();
	updateView();
}


Camera::~Camera()
{

}

__host__ void
Camera::translate(float x, float y, float z)
{
	position += glm::vec3(x, y, z);
	updateView();
}


void
Camera::constructViewProjection(glm::mat4& target)
{
	target = projection * view;
}

void
Camera::constructInverseViewProjection(glm::mat4& target)
{
	constructViewProjection(target);

	target = glm::inverse(target);
}

void
Camera::updateProjection()
{
	float aspectRatio = width / height;
	projection = glm::perspective(fieldOfView, aspectRatio, near, far);
}

void
Camera::updateView()
{
	//view = glm::mat4();
	view = glm::lookAt(position, position + glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
}

glm::mat4*
Camera::getProjection()
{
	return &projection;
}

glm::mat4*
Camera::getView()
{
	return &view;
}
