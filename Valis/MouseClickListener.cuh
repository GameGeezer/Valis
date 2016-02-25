#ifndef VALIS_MOUSE_LISTENER_CUH
#define VALIS_MOUSE_LISTENER_CUH

#include "GLLibraries.h"

enum MouseButton
{
	LEFT = GLFW_MOUSE_BUTTON_LEFT,
	RIGHT = GLFW_MOUSE_BUTTON_RIGHT
};

class MouseClickListener
{
public:
	virtual void
	onMouseRelease(MouseButton button, float posX, float posY) = 0;

	virtual void
	onMousePress(MouseButton button, float posX, float posY) = 0;
};


#endif //FERFUR_KEYBOARDLISTENER_H