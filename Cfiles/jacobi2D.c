/**
 * jacobi2D.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "jacobi2D.h"
#include "Polybench/polybench.h"
#include "Polybench/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif



#define TSTEPS 20
char str_temp[1024];
int N = 0;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



void read_cl_file(char* path)
{
	// Load the kernel source code into the array source_str
	char *pwd = strcat(path,"/Cfiles/jacobi2D.cl");
	fp = fopen(pwd, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i][j] = ((DATA_TYPE) i*(j+2) + 10) / N;
			B[i][j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
		}
	}
}


void cl_initialization(int plat, int dev)
{	

	cl_uint dev_cnt = 0;
  	clGetPlatformIDs(0, 0, &dev_cnt);
	
   	cl_platform_id platform_ids[100];
  	clGetPlatformIDs(dev_cnt, platform_ids, NULL);

   // Connect to a compute device

   	int num_devices = 0;
   	errcode = clGetDeviceIDs(platform_ids[plat], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
   	if (errcode != CL_SUCCESS)
   	{
       	printf("Error: Failed to create a device group!\n");
       	return EXIT_FAILURE;
  	}
   
   	cl_device_id * device_list = NULL;
   	device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
   	cl_int clStatus = clGetDeviceIDs(platform_ids[plat], CL_DEVICE_TYPE_ALL,num_devices,device_list,NULL);
   
   
   	char query[1024] ;
   	cl_int clError = clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME, sizeof(query), &query, NULL);
   	printf(" The program jacobi2D will execute on : %s \n", query);
  
   	// Create a compute context 
   	clGPUContext = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &errcode);
  
   	if (!clGPUContext)
   	{
       	printf("Error: Failed to create a compute context!\n");
       	return EXIT_FAILURE;
   	}

   // Create a command commands

   	clCommandQue = clCreateCommandQueue(clGPUContext, device_list[dev], CL_QUEUE_PROFILING_ENABLE, &errcode);
   	if (!clCommandQue)
   	{
       	printf("Error: Failed to create a command queue!\n");
       	return EXIT_FAILURE;
   	}
}

void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, N * N * sizeof(DATA_TYPE), NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, N * N * sizeof(DATA_TYPE), NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), B, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "runJacobi2D_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clKernel2 = clCreateKernel(clProgram, "runJacobi2D_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel1(int n)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = N;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel2(int n)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = N;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernels()
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = N;
	globalWorkSize[1] = N;
	int t;
	
	for (t = 0; t < TSTEPS ; t++)
	{	
		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
		clFinish(clCommandQue);

		// Execute the OpenCL kernel
		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
		clFinish(clCommandQue);
	}
}

void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
	int t, i, j;
	for (t = 0; t < _PB_TSTEPS; t++)
	{
    		for (i = 1; i < _PB_N - 1; i++)
		{
			for (j = 1; j < _PB_N - 1; j++)
			{
	  			B[i][j] = 0.2f * (A[i][j] + A[i][(j-1)] + A[i][(1+j)] + A[(1+i)][j] + A[(i-1)][j]);
			}
		}
		
    		for (i = 1; i < _PB_N-1; i++)
		{
			for (j = 1; j < _PB_N-1; j++)
			{
	  			A[i][j] = B[i][j];
			}
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}



void get_pwd(char* path, int n){

  FILE *fp;
  

  /* Open the command for reading. */
  fp = popen("pwd", "r");
  if (fp == NULL) {
    printf("Failed to run command\n" );
    exit(1);
  }
   

    int i = 0;
    char c = fgetc(fp);
    while ( c != EOF)
    {
	
	if (c == '\n') break;
        path[i++] = (char) c;
        c = fgetc(fp);
    }

    path[i] = '\0';  
    pclose(fp);

  
}

int main(int argc, char *argv[])
{
	char path[1000]; // current folder path
   	get_pwd(path,sizeof(path));
   	int plat = atoi(argv[1]);// which platform to run on
        int dev = atoi(argv[2]);// which device to run on

	N = atoi(argv[3]);
	/* Retrieve problem size. */
	int n = N;
	int tsteps = TSTEPS;

	POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

	read_cl_file(path);
	cl_initialization(plat,dev);
	cl_mem_init(POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
	cl_load_prog();
	
	/* Start timer. */
  	polybench_start_instruments;

	int t;
	for (t = 0; t < _PB_TSTEPS ; t++)
    	{
		cl_launch_kernel1(n);
		cl_launch_kernel2(n);
	}


	/* Stop and print timer. */
	printf("Execution Time in seconds: ");
  	polybench_stop_instruments;
 	double executionTime = polybench_print_instruments;
 	printf ("%0.6lf\n",executionTime);
 	FILE *myfile ;
   	myfile = fopen ("/home/taha/Videos/Doctorat/OpenCL/MySchedV2/dataSet/FeaturesExtractor/b2.txt","a");
   	fprintf(myfile,"%s","jacobi2D");
   	fprintf(myfile,"|");
   	fprintf(myfile,"%0.6lf",executionTime);
   	fprintf(myfile,"|");
   	fprintf(myfile,"%d",N*N+N*N);
   	fprintf(myfile,"\n");
   	fclose (myfile);
	
	errcode = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(a_outputFromGpu), 0, NULL, NULL);
	errcode = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, N * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(b_outputFromGpu), 0, NULL, NULL);
	
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");



	cl_clean_up();
	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(a_outputFromGpu);
	POLYBENCH_FREE_ARRAY(b);
	POLYBENCH_FREE_ARRAY(b_outputFromGpu);
    
	return 0;
}

#include "Polybench/polybench.c"
