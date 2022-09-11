/**
 * correlation.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <math.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "correlation.h"
#include "Polybench/polybench.h"
#include "Polybench/polybenchUtilFuncts.h"
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)


#define sqrt_of_array_cell(x,j) sqrt(x[j])

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

char str_temp[1024];
int N = 0, M= 0;

#define FLOAT_N 3214212.01
#define EPS 0.005

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_std;
cl_kernel clKernel_reduce;
cl_kernel clKernel_corr;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem stddev_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



void read_cl_file(char* path)
{
	// Load the kernel source code into the array source_str
	char *pwd = strcat(path,"/Cfiles/correlation.cl");
	fp = fopen(pwd, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}

void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
{
	int i, j;
	
	for (i=0; i < m; i++) 
	{
    		for (j=0; j < n; j++) 
		{
       		data[i][j] = ((DATA_TYPE) i*j)/ M;	
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
   	printf(" The program correlation will execute on : %s \n", query);
  
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



void cl_mem_init(DATA_TYPE POLYBENCH_2D(data,M,N,m,n), DATA_TYPE POLYBENCH_1D(mean,M,m), DATA_TYPE POLYBENCH_1D(stddev,M,m), DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))
{
	data_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	symmat_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &errcode);
	stddev_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M, NULL, &errcode);
	mean_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, data, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, symmat, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, stddev_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M, stddev, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M, mean, 0, NULL, NULL);
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
	clKernel_mean = clCreateKernel(clProgram, "mean_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");

	clKernel_std = clCreateKernel(clProgram, "std_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");

	clKernel_reduce = clCreateKernel(clProgram, "reduce_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel3\n");

	clKernel_corr = clCreateKernel(clProgram, "corr_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel4\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel(int m, int n)
{
	DATA_TYPE float_n = FLOAT_N;
	DATA_TYPE eps = EPS;

	size_t localWorkSize_Kernel1[2], globalWorkSize_Kernel1[2];
	size_t localWorkSize_Kernel2[2], globalWorkSize_Kernel2[2];
	size_t localWorkSize_Kernel3[2], globalWorkSize_Kernel3[2];
	size_t localWorkSize_Kernel4[2], globalWorkSize_Kernel4[2];

	localWorkSize_Kernel1[0] = DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	localWorkSize_Kernel1[1] = DIM_LOCAL_WORK_GROUP_KERNEL_1_Y;
	globalWorkSize_Kernel1[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_1_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_1_X;
	globalWorkSize_Kernel1[1] = 1;

	localWorkSize_Kernel2[0] = DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	localWorkSize_Kernel2[1] = DIM_LOCAL_WORK_GROUP_KERNEL_2_Y;
	globalWorkSize_Kernel2[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_2_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_2_X;
	globalWorkSize_Kernel2[1] = 1;

	localWorkSize_Kernel3[0] = DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	localWorkSize_Kernel3[1] = DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;
	globalWorkSize_Kernel3[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_X;
	globalWorkSize_Kernel3[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_3_Y)) * DIM_LOCAL_WORK_GROUP_KERNEL_3_Y;

	localWorkSize_Kernel4[0] = DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	localWorkSize_Kernel4[1] = DIM_LOCAL_WORK_GROUP_KERNEL_4_Y;
	globalWorkSize_Kernel4[0] = (size_t)ceil(((float)M) / ((float)DIM_LOCAL_WORK_GROUP_KERNEL_4_X)) * DIM_LOCAL_WORK_GROUP_KERNEL_4_X;
	globalWorkSize_Kernel4[1] = 1;

	/* Start timer. */
  	polybench_start_instruments;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_mean, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_mean, 3, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_mean, 4, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_std, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  clSetKernelArg(clKernel_std, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= clSetKernelArg(clKernel_std, 2, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_std, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_std, 4, sizeof(DATA_TYPE), (void *)&eps);
	errcode |= clSetKernelArg(clKernel_std, 5, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_std, 6, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments2\n");
 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_std, 1, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel2\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel_reduce, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	errcode =  clSetKernelArg(clKernel_reduce, 1, sizeof(cl_mem), (void *)&stddev_mem_obj);
	errcode |= clSetKernelArg(clKernel_reduce, 2, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_reduce, 3, sizeof(DATA_TYPE), (void *)&float_n);
	errcode |= clSetKernelArg(clKernel_reduce, 4, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_reduce, 5, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments3\n");
 
	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel3\n");
	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel	
	errcode =  clSetKernelArg(clKernel_corr, 0, sizeof(cl_mem), (void *)&symmat_mem_obj);
	errcode |= clSetKernelArg(clKernel_corr, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	errcode |= clSetKernelArg(clKernel_corr, 2, sizeof(int), (void *)&m);
	errcode |= clSetKernelArg(clKernel_corr, 3, sizeof(int), (void *)&n);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments4\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel_corr, 1, NULL, globalWorkSize_Kernel4, localWorkSize_Kernel4, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel4\n");
	clEnqueueBarrier(clCommandQue);

	DATA_TYPE val = 1.0;
	clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, ((M-1)*M + (M-1))*sizeof(DATA_TYPE), sizeof(DATA_TYPE), &val, 0, NULL, NULL);

	clFinish(clCommandQue);

	/* Stop and print timer. */
	printf("Execution Time in seconds: ");
  	polybench_stop_instruments;
 	double executionTime = polybench_print_instruments;
 	printf ("%0.6lf\n",executionTime);
 	FILE *myfile ;
   	myfile = fopen ("/home/taha/Videos/Doctorat/OpenCL/MySchedV2/dataSet/FeaturesExtractor/b2.txt","a");
   	fprintf(myfile,"%s","correlation");
   	fprintf(myfile,"|");
   	fprintf(myfile,"%0.6lf",executionTime);
   	fprintf(myfile,"|");
   	fprintf(myfile,"%d",M*N);
   	fprintf(myfile,"\n");
   	fclose (myfile);
}



void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel_reduce);
	errcode = clReleaseKernel(clKernel_mean);
	errcode = clReleaseKernel(clKernel_std);
	errcode = clReleaseKernel(clKernel_corr);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(symmat_mem_obj);
	errcode = clReleaseMemObject(data_mem_obj);
	errcode = clReleaseMemObject(mean_mem_obj);
	errcode = clReleaseMemObject(stddev_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
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
	M = atoi(argv[4]);
	

	/* Retrieve problem size. */
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  	POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,N,m,n);
  	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,N,m,n);
  	
	init_arrays(m, n, POLYBENCH_ARRAY(data));// 

	read_cl_file(path);
	cl_initialization(plat,dev);
	
	cl_mem_init(POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat_outputFromGpu));
	cl_load_prog();

	cl_launch_kernel(m, n);

	errcode = clEnqueueReadBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, M * N * sizeof(DATA_TYPE), POLYBENCH_ARRAY(symmat_outputFromGpu), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");



	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(stddev);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);

	cl_clean_up();
	
	return 0;
}

#include "Polybench/polybench.c"
