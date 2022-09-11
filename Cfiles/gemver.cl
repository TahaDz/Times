// Data size = 0
/**
 * gemver.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;


	
__kernel void gemver_kernel1(__global float *A, __global float *V1, __global float *V2, __global float *U1, __global float *U2, int n) 
{    
	int j = get_global_id(0);
	int i = get_global_id(1);
	
	if ((i < n) && (j < n))
	{
		A[i*n + j] += U1[i] * V1[j] + U2[i] * V2[j];
	}
}


__kernel void gemver_kernel2(__global float *A, __global float *X, __global float *Y, __global float *Z, float beta, int n) 
{    
	int i = get_global_id(0);

	if (i < n)
	{
		int j;
		for(j = 0; j < n; j++) 
		{
			X[i] += beta * A[j * n + i] * Y[j];
		}
		X[i] += Z[i];
	}
}


__kernel void gemver_kernel3(__global float *A, __global float *X, __global float *w, float alpha, int n) 
{    
	int i = get_global_id(0);
	
	if (i < n)
	{
		int j;
		for(j = 0; j < n; j++)
		{ 
			w[i] += alpha * A[i*n + j] * X[j];
		}
	}
}
