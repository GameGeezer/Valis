#ifndef VALIS_SCREEN_H
#define VALIS_SCREEN_H

class Screen
{
	friend class Game;
protected:

	virtual void
	onCreate() = 0;

	virtual void
	onPause() = 0;

	virtual void
	onLeave() = 0;

	virtual void
	onResume() = 0;

	virtual void
	onUpdate(int delta) = 0;

	virtual void
	onResize(int width, int height) = 0;

	virtual void
	onDestroy() = 0;

private:
	bool isCreated;
};

#endif