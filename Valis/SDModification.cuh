
#ifndef VALIS_SD_MODIFICATION_CUH
#define VALIS_SD_MODIFICATION_CUH

#include "device_launch_parameters.h"

class SDModification
{
public:
	const int functionId;

	SDModification(int functionId) : functionId(functionId)
	{

	}

	__host__ virtual inline SDModification*
	copyToDevice() = 0;

};

#endif
