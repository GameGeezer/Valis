#ifndef VALIS_CUDAHELPER_CUH
#define VALIS_CUDAHELPER_CUH

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

#define assertCUDA(ans) { checkCUDAForError(ans, __FILE__, __LINE__); }
inline void checkCUDAForError(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

		if (abort)
		{
			exit(code);
		}
	}
}
#endif
