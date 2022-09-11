////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>
#include<time.h>
/*///////////////////////////////////////////////////////////////////////////////
#define WA 1024
#define HA 1024

#define WC HA
#define HC WA
///////////////////////////////////////////////////////////////////////////////*/
#define POLYBENCH_TIME 1
#include "Polybench/polybench.h"
#include "Polybench/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)


#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


int WA = 0;
int HA = 0;
int WC = 0;
int HC = 0;


// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
   int i;

   for (i = 0; i < size; ++i)
   	data[i] = rand() / (float)RAND_MAX;
}

long LoadOpenCLKernel(char const* path, char **buf)
{
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char *) malloc( fsz+1);
    if( NULL == *buf ) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }


    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';

    /* Return the file size */
    return (long)fsz;
}


// Get current file path 

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


int main(int argc, char** argv)
{


   WA = atoi(argv[3]);
   HA = atoi(argv[4]);
   
   WC = HA;
   HC = WA;
	
   char path[1000]; // current folder path
   char filename[100];
   get_pwd(path,sizeof(path));
   strcpy(filename,argv[0]);
   // argv[0] = Cfiles/2mm => remove Cfile/ => 2mm
   get_filename(filename);
   printf("%s\n",filename); 
   

   int err;                            // error code returned from api calls

   cl_device_id device_id;             // compute device id 
   cl_context context;                 // compute context
   cl_command_queue commands;          // compute command queue
   cl_program program;                 // compute program
   cl_kernel kernel;                   // compute kernel

    // OpenCL device memory for matrices
   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;

   // set seed for rand()
   srand(2014);
 
   //Allocate host memory for matrices A and B
   unsigned int size_A = WA * HA;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = size_A;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);

   //Initialize host memory
   randomMemInit(h_A, size_A);
  /* 
   for(int i = 0; i < size_A; i++)
   {
      printf("%f ", h_A[i]);
      if(((i + 1) % WA) == 0)
      printf("\n");
   }
   printf("\n");
   */
   for (int i = 0; i < size_A; i++) h_B[i] = 0;
 
   //Allocate host memory for the result C
   unsigned int size_C = WC * HC;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C = (float*) malloc(mem_size_C);
  
   printf("Initializing OpenCL device...\n"); 

   cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);
	
   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
   
   int plat = atoi(argv[1]);// which platform to run on
   int dev = atoi(argv[2]);// which device to run on
	
   // Connect to a compute device

   int num_devices = 0;
   err = clGetDeviceIDs(platform_ids[plat], CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to create a device group!\n");
       return EXIT_FAILURE;
   }
   
   cl_device_id * device_list = NULL;
   device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
   cl_int clStatus = clGetDeviceIDs(platform_ids[plat], CL_DEVICE_TYPE_ALL,num_devices,device_list,NULL);
   
   
   char query[1024] ;
   cl_int clError = clGetDeviceInfo(device_list[dev], CL_DEVICE_NAME, sizeof(query), &query, NULL);
   printf(" The program matrix transpose will execute on : %s \n", query);
  
   // Create a compute context 
   context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &err);
  
  
   if (!context)
   {
       printf("Error: Failed to create a compute context!\n");
       return EXIT_FAILURE;
   }

   // Create a command commands

   commands = clCreateCommandQueue(context, device_list[dev], CL_QUEUE_PROFILING_ENABLE, &err);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       return EXIT_FAILURE;
   }

   // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;

   char *pwd = strcat(path,"/Cfiles/transpose.cl");// kernel absolute path (starting from the root element )
   lFileSize = LoadOpenCLKernel(pwd, &KernelSource);
   if( lFileSize < 0L ) {
       perror("File read failed");
       return 1;
   }

   program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   if (!program)
   {
       printf("Error: Failed to create compute program!\n");
       return EXIT_FAILURE;
   }

   // Build the program executable
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];
       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       exit(1);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, "transpose", &err);
   if (!kernel || err != CL_SUCCESS)
   {
       printf("Error: Failed to create compute kernel!\n");
       exit(1);
   }

   // Create the input and output arrays in device memory for our calculation
   d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
   d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &err);
   d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &err);

   if (!d_A || !d_B || !d_C)
   {
       printf("Error: Failed to allocate device memory!\n");
       exit(1);
   }    
    
   //printf("Running matrix traspose for matrices A (%dx%d) \n", WA,HA); 

   //Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
   
   
 
   int wA = WA;
   int wC = WC;
   
   
   //struct timespec begin, end; 
   //clock_gettime(CLOCK_REALTIME, &begin);
   
   polybench_start_instruments;
	
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_A);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_C);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA);
   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to set kernel arguments! %d\n", err);
       exit(1);
   }
 
   localWorkSize[0] = 16;
   localWorkSize[1] = 16;
   globalWorkSize[0] = 1024;
   globalWorkSize[1] = 1024;
   
   
   //cl_event event;
 
   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL,NULL);// &event);
   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to execute kernel! %d\n", err);
       exit(1);
   }
   
   //clWaitForEvents(1, &event);
   clFinish(commands);
   polybench_stop_instruments;
   //clock_gettime(CLOCK_REALTIME, &end);
   //double elapsed = end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec)*1e-9;

    //long seconds = end.tv_sec - begin.tv_sec;
    //long nanoseconds = end.tv_nsec - begin.tv_nsec;
    
   //printf("Time measured: %0.9f seconds.\n", elapsed);
  // cl_ulong time_start;
  // cl_ulong time_end;
   //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
   //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
   
   double executionTime =  polybench_print_instruments;
   //printf ("%0.6lf\n",t);
   
   //double executionTime = (double) (stop - start) / CLOCKS_PER_SEC;
   //printf("Runtime of transpose: %0.6f seconds\n", executionTime);
   //double nanoSeconds = time_end-time_start;
   //printf("OpenCl Execution time is: %0.6f seconds \n",nanoSeconds / 1000000000.0);
   FILE *myfile ;
   myfile = fopen ("/home/taha/Videos/Doctorat/OpenCL/MySchedV2/dataSet/FeaturesExtractor/b2.txt","a");
   fprintf(myfile,"%s","transpose");
   fprintf(myfile,"|");
   fprintf(myfile,"%0.6lf",executionTime);
   fprintf(myfile,"|");
   fprintf(myfile,"%d",WA*HA);
   fprintf(myfile,"\n");
   fclose (myfile); 

   
 
   //Retrieve result from device
   err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to read output array! %d\n", err);
       exit(1);
   }
 
   //print out the results
/*
   printf("\n\nMatrix C (Results)\n");
   
   for(int i = 0; i < size_C; i++)
   {
      printf("%f ", h_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");
*/
  
  // printf("Matrix trasposition completed...\n"); 

   //Shutdown and cleanup
   free(h_A);
   free(h_B);
   free(h_C);
 
   clReleaseMemObject(d_A);
   clReleaseMemObject(d_C);
   clReleaseMemObject(d_B);

   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(commands);
   clReleaseContext(context);

   return 0;
}

#include "Polybench/polybench.c"
