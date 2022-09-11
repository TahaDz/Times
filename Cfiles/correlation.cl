/**
 * correlation.cl: This file is part of the PolyBench/GPU 1.0 test suite.
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


__kernel void mean_kernel(__global float *mean, __global float *data, float float_n, int m, int n) 
{    
	int j = get_global_id(0);
	
	if (j < m)
	{
		mean[j] = 0.0;

		int i;
		for (i=0; i < n; i++)
		{
			mean[j] += data[i*m + j];
		}
		
		mean[j] /= (float)float_n;
	}
}


__kernel void std_kernel(__global float *mean, __global float *std, __global float *data, float float_n, float eps, int m, int n) 
{
	int j = get_global_id(0);

	if (j < m)
	{
		std[j] = 0.0;

		int i;
		for (i = 0; i < n; i++)
		{
			std[j] += (data[i*m + j] - mean[j]) * (data[i*m + j] - mean[j]);
		}
		std[j] /= float_n;
		std[j] =  sqrt(std[j]);
		if(std[j] <= eps) 
		{
			std[j] = 1.0;
		}
	}
}


__kernel void reduce_kernel(__global float *mean, __global float *std, __global float *data, float float_n, int m, int n) 
{
	int j = get_global_id(0);
	int i = get_global_id(1);
	
	if ((i < n) && (j < m))
	{
		data[i*m + j] -= mean[j];
		data[i*m + j] /= (sqrt(float_n) * std[j]);
	}
}


__kernel void corr_kernel(__global float *symmat, __global float *data, int m, int n) 
{
	int j1 = get_global_id(0);
	
	int i, j2;
	if (j1 < (m-1))
	{
		symmat[j1*m + j1] = 1.0;

		for (j2 = (j1 + 1); j2 < m; j2++)
		{
			for(i = 0; i < n; i++)
			{
				symmat[j1*m + j2] += data[i*m + j1] * data[i*m + j2];
			}
			symmat[j2*m + j1] = symmat[j1*m + j2];
		}
	}
}



