#ifndef VALIS_CAMERA_H
#define VALIS_CAMERA_H

#include "device_launch_parameters.h"
#include <glm/mat4x4.hpp>
#include <glm\vec3.hpp>

class Camera
{
public:
	Camera(float width, float height, float near, float far, float fieldOfView);
	~Camera();
	
	__host__ void
	moveForward(float amount);

	__host__ void
	moveBackward(float amount);

	__host__ void
	moveLeft(float amount);

	__host__ void
	moveRight(float amount);

	__host__ void
	translate(float x, float y, float z);

	__host__ void
	rotate(float ammount, glm::vec3 axis);

	__host__ void
	rotateLocalX(float ammount);

	__host__ void
	constructViewProjection(glm::mat4& target);

	__host__ void
	constructInverseViewProjection(glm::mat4& target);

	__host__ glm::mat4*
	getProjection();

	__host__ glm::mat4*
	getView();

private:
	__host__ void
	updateProjection();

	__host__ void
	updateView();

	float width, height, near, far, fieldOfView;
	glm::mat4 projection, view;
	glm::vec3 position, direction, up;
};

#endif