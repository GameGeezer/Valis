#ifndef VALIS_PLAYER_CUH
#define VALIS_PLAYER_CUH

#include <glm\vec2.hpp>
#include <glm\vec3.hpp>
#include <glm\mat4x4.hpp>
#include <chrono>

#include "KeyboardListener.cuh"
#include "MouseClickListener.cuh"
#include "MouseMoveListener.cuh"

class Camera;
class SDFDevice;
class SDFHost;
class PlaceSDPrimitive;
class CarveSDPrimitive;
class BlendSDModification;
class SDModification;

using namespace std::chrono;

class Player : public KeyboardListener, public MouseClickListener, public MouseMoveListener
{
public:
	Camera* camera;
	SDFDevice* deviceEditSDF;

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
	bool isLeftMousePressed = false, isRightMousePressed = false, isWPressed = false, isAPressed = false, isSPressed = false, isDPressed = false, isKPressed = false, isLPressed = false, isOPressed = false, isPPressed = false, isQPressed = false, isEPressed = false;

	glm::vec3 scale;
	
	SDFHost* hostEditSDF;
	PlaceSDPrimitive* place;
	CarveSDPrimitive* carve;
	BlendSDModification* blend;
	SDModification * currentMod;
	milliseconds lastPlaceTime;
	glm::vec3 rotation;
	glm::mat4 orientation;
	float distanceFromCamera = 0.5f;
	int brushType = 1;
};

#endif