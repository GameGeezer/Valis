#ifndef VALIS_PLAYER_CUH
#define VALIS_PLAYER_CUH

#include "KeyboardListener.cuh"
#include "MouseClickListener.cuh"

class Camera;

class Player : public KeyboardListener, public MouseClickListener
{
public:
	Camera* camera;

	Player(Camera& camera);

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
};

#endif