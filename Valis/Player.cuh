#ifndef VALIS_PLAYER_CUH
#define VALIS_PLAYER_CUH

#include "KeyboardListener.cuh"

class Camera;

class Player : public KeyboardListener
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
};

#endif