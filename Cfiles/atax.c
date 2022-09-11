/**
 * atax.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif



#define POLYBENCH_TIME 1

//select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_GPU

#include "atax.h"
#include "Polybench/polybench.h"
#include "Polybench/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

#ifndef M_PI
#define M_PI 3.14159
#endif

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


char str_temp[1024];
int NX = 0;
int NY = 0;


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
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem tmp_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;

#define RUN_ON_CPU





void read_cl_file(char* path)
{
	// Load the kernel source code into the array source_str
	char *pwd = strcat(path,"/Cfiles/atax.cl");
	fp = fopen(pwd, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}

void init_array(int nx, int ny, DATA_TYPE POLYBENCH_1D(x,NX,nx), DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny))
{
	int i, j;

	for (i = 0; i < nx; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < ny; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NX;
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
   	printf(" The program atax will execute on : %s \n", query);
  
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

void cl_mem_init(DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NY,ny), DATA_TYPE POLYBENCH_1D(y,NY,ny), DATA_TYPE POLYBENCH_1D(tmp,NX,nx))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &errcode);
	x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NY, NULL, &errcode);
	y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NY, NULL, &errcode);
	tmp_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");
	
	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NY, x, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NY, y, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, tmp_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX, tmp, 0, NULL, NULL);
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
		
	// Create the 1st OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "atax_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

	// Create the 2nd OpenCL kernel
	clKernel2 = clCreateKernel(clProgram, "atax_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel(int nx, int ny,char* filename)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NX) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	/* Start timer. */
  	polybench_start_instruments;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&x_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&tmp_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&nx);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&ny);
	if(errcode != CL_SUCCESS) printf("Error in setting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clEnqueueBarrier(clCommandQue);
	
	globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = 1;

	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&y_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&tmp_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&nx);
	errcode |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&ny);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);

	/* Stop and print timer. */
	printf("Execution Time in seconds:");
  	polybench_stop_instruments;
 	double executionTime = polybench_print_instruments;
 	printf ("%0.6lf\n",executionTime);
 	FILE *myfile ; 
	myfile = fopen ("/home/taha/Videos/Doctorat/OpenCL/MySchedV2/dataSet/FeaturesExtractor/b2.txt","a");
   	fprintf(myfile,"%s","atax");
   	fprintf(myfile,"|");
   	fprintf(myfile,"%0.6lf",executionTime);
   	fprintf(myfile,"|");
   	fprintf(myfile,"%d",NX*NY+NY);
   	fprintf(myfile,"\n");
   	fclose (myfile);

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
	errcode = clReleaseMemObject(x_mem_obj);
	errcode = clReleaseMemObject(y_mem_obj);
	errcode = clReleaseMemObject(tmp_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

/*
void atax_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NY,ny), DATA_TYPE POLYBENCH_1D(y,NY,ny), 
		DATA_TYPE POLYBENCH_1D(tmp,NX,nx))
{
	int i,j;
	
	for (i= 0; i < _PB_NY; i++)
	{
    		y[i] = 0;
	}
  
	for (i = 0; i < _PB_NX; i++)
 	{
      		tmp[i] = 0;

      		for (j = 0; j < _PB_NY; j++)
		{
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
		
      		for (j = 0; j < _PB_NY; j++)
		{
			y[j] = y[j] + A[i][j] * tmp[i];
		}
    }
}

*/
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx, DATA_TYPE POLYBENCH_1D(y,NX,nx))
{
  int i;

  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
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

void get_filename(char* filename){

		for (int i = 0; i<strlen(filename);i++){ 
		if (filename[i] != '/') i++;
		else{
			int j = 0;
			i++;
			for (; j < strlen(filename)&&i<strlen(filename);j++) {
				filename[j] = filename[i];
				i++;
			}
			filename[j] = '\0';
			break;
		}	
	}

}


int main(int argc, char *argv[])
{

	
	char path[1000]; // current folder path
	char filename[100];
   	get_pwd(path,sizeof(path));
   	strcpy(filename,argv[0]);
	// argv[0] = Cfiles/2mm => remove Cfile/ => 2mm
	get_filename(filename);

	printf("%s\n",filename);	
	
   	
   	int plat = atoi(argv[1]);// which platform to run on
	int dev = atoi(argv[2]);// which device to run on
	
	NX = atoi(argv[3]);
	NY = atoi(argv[4]);
	int nx = NX;
	int ny = NY;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny); 
	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(A));

	read_cl_file(path);
	cl_initialization(plat,dev);
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp));
	cl_load_prog();

	cl_launch_kernel(nx, ny,filename);

	errcode = clEnqueueReadBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0, NY*sizeof(DATA_TYPE), POLYBENCH_ARRAY(y_outputFromGpu), 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

/*
	#ifdef RUN_ON_CPU

		// Start timer. 
	  	polybench_start_instruments;

		atax_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp));
	
		// Stop and print timer.
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ny, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ny, POLYBENCH_ARRAY(y_outputFromGpu)));

	#endif //RUN_ON_CPU*/

	cl_clean_up();
	
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(x);
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);
	POLYBENCH_FREE_ARRAY(tmp);
	
	return 0;
}

#include "Polybench/polybench.c"
