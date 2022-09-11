/**
 * doitgen.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include<math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif




#define POLYBENCH_TIME 1

#include "doitgen.h"
#include "Polybench/polybench.h"
#include "Polybench/polybenchUtilFuncts.h"
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU

#define MAX_SOURCE_SIZE (0x100000)


char str_temp[1024];
int NR = 0;
int NQ = 0;
int NP = 0;

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


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*void kernel_doitgenCpu(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np))
{
	int r, q, p, s;

	for (r = 0; r < _PB_NR; r++)
	{
		for (q = 0; q < _PB_NQ; q++)  
		{
			for (p = 0; p < _PB_NP; p++)  
			{
				sum[r][q][p] = 0;
				for (s = 0; s < _PB_NP; s++)
					sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
			}
			for (p = 0; p < _PB_NR; p++)
				A[r][q][p] = sum[r][q][p];
		}
	}

}

*/

/* Array initialization. */
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
	int i, j, k;

	for (i = 0; i < nr; i++)
		for (j = 0; j < nq; j++)
			for (k = 0; k < np; k++)
				A[i][j][k] = ((DATA_TYPE) i*j + k) / np;

	for (i = 0; i < np; i++)
		for (j = 0; j < np; j++)
			C4[i][j] = ((DATA_TYPE) i*j) / np;
}







void read_cl_file(char* path)
{
	// Load the kernel source code into the array source_str
	char *pwd = strcat(path,"/Cfiles/doitgen.cl");
	fp = fopen(pwd, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
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
   	printf(" The program doitgen will execute on : %s \n", query);
  
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
void cl_mem_init(DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np))
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NR * NQ * NP * sizeof(DATA_TYPE), NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NP * NP * sizeof(DATA_TYPE), NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NR * NQ * NP * sizeof(DATA_TYPE), NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, NR * NQ * NP * sizeof(DATA_TYPE), A, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, NP * NP * sizeof(DATA_TYPE), C4, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, NR * NQ * NP * sizeof(DATA_TYPE), sum, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "doitgen_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clKernel2 = clCreateKernel(clProgram, "doitgen_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel1(int nr, int nq, int np, int r)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NP) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NQ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(int), (void *)&nr);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(int), (void *)&nq);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(int), (void *)&np);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 6, sizeof(int), (void *)&r);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
}

void cl_launch_kernel2(int nr, int nq, int np, int r)
{
	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NP) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NQ) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel2, 0, sizeof(int), (void *)&nr);
	errcode |= clSetKernelArg(clKernel2, 1, sizeof(int), (void *)&nq);
	errcode |= clSetKernelArg(clKernel2, 2, sizeof(int), (void *)&np);
	errcode |= clSetKernelArg(clKernel2, 3, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 4, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 5, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel2, 6, sizeof(int), (void *)&r);
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);
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
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
	int i, j, k;

	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < nq; j++)
		{
			for (k = 0; k < np; k++) 
			{
				fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j][k]);
				if (i % 20 == 0) fprintf (stderr, "\n");
			}
		}
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
	char filename[100];
	strcpy(filename,argv[0]);
	// argv[0] = Cfiles/2mm => remove Cfile/ => 2mm
	get_filename(filename);

	printf("%s\n",filename);	
	
	char path[1000]; // current folder path
   	get_pwd(path,sizeof(path));
   	int plat = atoi(argv[1]);// which platform to run on
        int dev = atoi(argv[2]);// which device to run on
        NR = atoi(argv[3]);
	NQ = atoi(argv[4]);
	NP = atoi(argv[5]);

	/* Retrieve problem size. */
	int nr = NR;
	int nq = NQ;
	int np = NP;

	/* Variable declaration/allocation. */
	POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_3D_ARRAY_DECL(sum,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_3D_ARRAY_DECL(sum_outputFromGpu,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np); 

	/* Initialize array(s). */
	init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

	read_cl_file(path);
	cl_initialization(plat,dev);
	cl_mem_init(POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4), POLYBENCH_ARRAY(sum));
	cl_load_prog();

	/* Start timer. */
	polybench_start_instruments;
  	


	int r;
	for (r = 0; r < NR; r++)
	{
		cl_launch_kernel1(nr, nq, np, r);
		cl_launch_kernel2(nr, nq, np, r);
	}

	/* Stop and print timer. */
	printf("Execution Time in seconds:");
  	polybench_stop_instruments;
 	double executionTime = polybench_print_instruments;
 	printf ("%0.6lf\n",executionTime);
 	FILE *myfile ; 
   	myfile = fopen ("/home/taha/Videos/Doctorat/OpenCL/MySchedV2/dataSet/FeaturesExtractor/b2.txt","a");
   	fprintf(myfile,"%s","doitgen");
   	fprintf(myfile,"|");
   	fprintf(myfile,"%0.6lf",executionTime);
   	fprintf(myfile,"|");
   	fprintf(myfile,"%d",NR*NQ*NP+ NP*NP);
   	fprintf(myfile,"\n");
  	fclose (myfile); 

	errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, NR * NQ * NP * sizeof(DATA_TYPE), sum_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");



	cl_clean_up();

	/* Garbage collection */
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(sum);
	POLYBENCH_FREE_ARRAY(sum_outputFromGpu);
	POLYBENCH_FREE_ARRAY(C4);	

	return 0;
}

#include "Polybench/polybench.c"
