#ifndef VALIS_KEYBOARDLISTENER_H
#define VALIS_KEYBOARDLISTENER_H

class KeyboardListener
{
public:
	virtual void
	onKeyRelease(int keyCode) = 0;

	virtual void
	onKeyPress(int keyCode) = 0;

	virtual void
	onKeyRepeat(int keyCode) = 0;
};


#endif //FERFUR_KEYBOARDLISTENER_H