#ifndef VALIS_PLAYER_CUH
#define VALIS_PLAYER_CUH

#include <glm\vec2.hpp>

#include "KeyboardListener.cuh"
#include "MouseClickListener.cuh"
#include "MouseMoveListener.cuh"

class Camera;

class Player : public KeyboardListener, public MouseClickListener, public MouseMoveListener
{
public:
	Camera* camera;

	Player(Camera& camera);

	void
	update(int delta);

	void
	onKeyRelease(int keyCode) override;

	void
	onKeyPress(int keyCode) override;

	void
	onKeyRepeat(int keyCode) override;

	void
	onMouseRelease(MouseButton button, float posX, float posY) override;

	void
	onMousePress(MouseButton button, float posX, float posY) override;

	void
	onMouseMove(float posX, float posY) override;

private:
	glm::vec2 lastMousePosition, lastMouseClickPosition;
	bool isMousePressed = false, isWPressed = false, isAPressed = false, isSPressed = false, isDPressed = false;
};

#endif