#ifndef COMMON_H
#define COMMON_H

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#include <cuda_runtime.h>
#include <cinder/app/AppBase.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		cinder::app::console() << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
		//fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif