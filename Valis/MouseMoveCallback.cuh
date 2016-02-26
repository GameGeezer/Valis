#ifndef VALIS_MOUSE_MOVE_CALLBACK_CUH
#define VALIS_MOUSE_MOVE_CALLBACK_CUH

#include <vector>

class MouseMoveListener;
class GLFWwindow;

using namespace std;

class MouseMoveCallback
{
public:

	MouseMoveCallback();

	void
	invoke(float xpos, float ypos);

	void
	addListener(MouseMoveListener& listener);

	void
	removeListener(MouseMoveListener& listener);

	void
	clearListeners();

private:
	vector<MouseMoveListener*> listeners;
};


#endif //VALIS_KEYBOARDCALLBACK_H