/////////////////////////////////////////////
///                                       ///
///     Author : Stéphane Küng            ///
///       Date : 10.12.27015              ///
///     Lesson : HPC                      ///
///   Language : CPP                      ///
///                                       ///
/////////////////////////////////////////////
///
///      Linux : g++ vectAddCL.cpp -I/usr/local/cuda/include -L/usr/lib/nvidia-340 -lm -lOpenCL -o vectAddCL
///      MACOS : g++ vectAddCL.cpp -lm -framework OpenCL -o vect
///
/////////////////////////////////////////////


/// Includes ///

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __MACH__
#include <OpenCL/opencl.h>
#endif

#ifdef __linux__
#include <CL/opencl.h>
#endif

#include <time.h>


/// Functions ///

int showDeviceInfo();
int VectorCPU(long n);
int VectorOpenCL(long n);
int MandelbrotCPU(int w, int h);
int MandelbrotOpenCL(int w, int h);
int main( int argc, char* argv[] );


// OpenCL kernel. Each work item takes care of one element of c
const char *vectorKernelSource =                                "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   \n" \
"#pragma OPENCL EXTENSION map_f64_to_f32 : enable                \n" \
"__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n) {												 \n" \
"       c[id] = 0;                                               \n" \
"		for (int x = 0; x < 100; x++)                            \n" \
"        	c[id] += (a[id] + b[id]);                            \n" \
"                                                                \n" \
"       c[id] = c[id] / 100;                                     \n" \
"    }                                                           \n" \
"}                                                               \n" \
																"\n" ;
/// Print informations about the OpenCL Device
int showDeviceInfo() {

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID

    char device_string[1024];

    // Bind to platform
    if (clGetPlatformIDs(1, &cpPlatform, NULL) != CL_SUCCESS) {
        printf("Platform error\n");
        return 0;
    }

    // Get ID for the device
    if (clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL) != CL_SUCCESS) {
        printf("Device error\n");
        return 0;
    }

    // CL_DEVICE_NAME
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf("CL_DEVICE_NAME: \t%s\n", device_string);
    
    // CL_DEVICE_VENDOR
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    printf("CL_DEVICE_VENDOR: \t%s\n", device_string);
    
    // CL_DRIVER_VERSION
    clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
    printf("CL_DRIVER_VERSION: \t%s\n", device_string);
    
    // CL_DEVICE_VERSION
    clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(device_string), &device_string, NULL);
    printf("CL_DEVICE_VERSION: \t%s\n\n", device_string);

    return 0;

}

int VectorOpenCL(long n) {
	char build_c[4096];
 
    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;
 
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(float);
 
    // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
 
    // Initialize vectors on host
    long i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = sinf(i)*sinf(i);
        h_b[i] = cosf(i)*cosf(i);
        h_c[i] = 0;
    }

    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 256;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;

    // Bind to platform
    if (clGetPlatformIDs(1, &cpPlatform, NULL) != CL_SUCCESS) {
        printf("Platform error\n");
    }

    // Get ID for the device
    if (clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL) != CL_SUCCESS) {
        printf("Device error\n");
    }

 
    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("Context error %d \n", err);
    }
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err != CL_SUCCESS) {
        printf("Command queue error %d \n", err);
    }
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & vectorKernelSource, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("clCreateProgramWithSource error %d \n", err);
    }
    // Build the program executable 
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS ) {
        printf( "Error on buildProgram  %d \n", err);
        printf("\n Error number %d", err);
        fprintf( stdout, "\nRequestingInfo\n" );
        clGetProgramBuildInfo( program, device_id, CL_PROGRAM_BUILD_LOG, 4096, build_c, NULL );
        printf( "Build Log for %s_program:\n%s\n", "example", build_c );
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
    if(err != CL_SUCCESS) {
        printf("clCreateKernel error %d \n", err);
    }
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) { printf("clCreateBuffer Error %d \n", err); }
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) { printf("clCreateBuffer Error %d \n", err); }
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) { printf("clCreateBuffer Error %d \n", err); }
 
    // Write our data set into the input array in device memory
    err  = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("clEnqueueWriteBuffer error %d \n", err);
    }

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
    if(err != CL_SUCCESS) {
        printf("clSetKernelArg error %d \n", err);
    }
    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("clEnqueueNDRangeKernel error %d \n", err);
    }
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    float sum = 0;
    for(i=0; i<n; i++) {
        //printf("intermediate c: %f \n", h_c[i]);
        sum += h_c[i];
    }
    printf("final result: %f \n", sum*1.0/n);
 
    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return sum*1.0/n;
}    

int VectorCPU(long n) {    

    // Input vectors
    double *a;
    double *b;
    // Output vector
    double *c;
     
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
     
    // Allocate memory for each vector
    a = (double*)malloc(bytes);
    b = (double*)malloc(bytes);
    c = (double*)malloc(bytes);
     
    // Initialize content of input vectors, vector a[i] = sin(i)^2 vector b[i] = cos(i)^2
    long i;
    for(i=0; i<n; i++) {
        a[i] = sin(i)*sin(i);
        b[i] = cos(i)*cos(i);
    }
     
    // Sum component wise and save result into vector c
    for(i=0; i<n; i++){
        //c[i] = a[i] + b[i];
        int x;
        for (x = 0; x< 100; x++) 
            c[i] += (((a[i] + b[i]) * (a[i] + b[i])) * ((a[i] + b[i]) * (a[i] + b[i]))) / (((a[i] + b[i]) * (a[i] + b[i])) * ((a[i] + b[i]) * (a[i] + b[i])));
        c[i] /= 100.0;
    }
     
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++) {
        sum += c[i];
    }
    sum = sum/n;
    printf("final result: %f\n", sum);
     
    // Release memory
    free(a);
    free(b);
    free(c);

    return 0;
}

int MandelbrotCPU(int w, int h){
    
    long sum = 0;

    //each iteration, it calculates: new = old*old + c, where c is a constant and old starts at current pixel
    float cRe, cIm;                   //real and imaginary part of the constant c, determinate shape of the Julia Set
    float newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old
    float zoom = 1, moveX = 0, moveY = 0; //you can change these to zoom and change position
    int maxIterations = 300; //after how much iterations the function should stop

    //pick some values for the constant c, this determines the shape of the Julia Set
    cRe = -0.7;
    cIm = 0.27015;

    //loop through every pixel
    for(int x = 0; x < w; x++)
    for(int y = 0; y < h; y++)
    {
        //calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
        newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
        newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;
        //i will represent the number of iterations
        int i;
        //start the iteration process
        for(i = 0; i < maxIterations; i++)
        {
            //remember value of previous iteration
            oldRe = newRe;
            oldIm = newIm;
            //the actual iteration, the real and imaginary part are calculated
            newRe = oldRe * oldRe - oldIm * oldIm + cRe;
            newIm = 2 * oldRe * oldIm + cIm;
            //if the point is outside the circle with radius 2: stop
            if((newRe * newRe + newIm * newIm) > 4) break;
        }

        //use color model conversion to get rainbow palette, make brightness black if maxIterations reached
        //i is color for x and y
        //if (i > 10){
        //    printf("%5d-%2d-%2d:",i,x,y);
        //}
        sum += i;

    }

    printf("final result: %ld \n",sum);
    return sum;

}

/// Mandelbrot Kernel
const char *MandelbrotKernelSource =                             "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global int *c,                          \n" \
"                       const int w,                              \n" \
"                       const int h,                              \n" \
"                       const long n)                             \n" \
"{                                                                \n" \
"    //Get our global thread ID                                   \n" \
"    int id = get_global_id(0);                                   \n" \
"    float cRe, cIm, newRe, newIm, oldRe, oldIm, zoom = 1, moveX = 0, moveY = 0; \n" \
"    int maxIterations = 300;                                     \n" \
"                                                                 \n" \
"    cRe = -0.7;                                                  \n" \
"    cIm = 0.27015;                                               \n" \
"                                                                 \n" \
"    int y = id % h;                                              \n" \
"    int x = (id-y)/h;                                            \n" \
"                                                                 \n" \
"    newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;        \n" \
"    newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;              \n" \
"                                                                 \n" \
"    //Make sure we do not go out of bounds                       \n" \
"    if (id < n) {                                                \n" \
"         int i;                                                  \n" \
"                                                                 \n" \
"         for(i = 0; i < maxIterations; i++) {                    \n" \
"               oldRe = newRe;                                    \n" \
"               oldIm = newIm;                                    \n" \
"                                                                 \n" \
"               newRe = oldRe * oldRe - oldIm * oldIm + cRe;      \n" \
"               newIm = 2 * oldRe * oldIm + cIm;                  \n" \
"                                                                 \n" \
"               if((newRe * newRe + newIm * newIm) > 4) break;    \n" \
"         }                                                       \n" \
"                                                                 \n" \
"         c[id] = i;                                              \n" \
"    }                                                            \n" \
"}                                                                \n" \
                                                                 "\n" ;

/// Use OpenCL device to calcul the sum of Mandelbrot
int MandelbrotOpenCL(int w, int h){

    long n = w * h;
    long sum = 0;

    char build_c[4096];
 
    // Host output vector
    int *h_c;
    // Device output buffer
    cl_mem d_c;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);
 
    // Allocate memory for each vector on host
    h_c = (int*)malloc(bytes);
 
    // Initialize vectors on host
    long i;
    for( i = 0; i < n; i++ )
    {
        h_c[i] = 0;
    }

    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 256;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;

    // Bind to platform
    if (clGetPlatformIDs(1, &cpPlatform, NULL) != CL_SUCCESS) {
        printf("Platform error\n");
    }
 
    // Get ID for the device
    if (clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL) != CL_SUCCESS) {
        printf("Device error\n");
    }
 
    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("Context error %d \n", err);
    }
    // Create a command queue 
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err != CL_SUCCESS) {
        printf("Command queue error %d \n", err);
    }
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & MandelbrotKernelSource, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("clCreateProgramWithSource error %d \n", err);
    }
    // Build the program executable 
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS ) {
        printf( "Error on buildProgram  %d \n", err);
        printf("\n Error number %d", err);
        fprintf( stdout, "\nRequestingInfo\n" );
        clGetProgramBuildInfo( program, device_id, CL_PROGRAM_BUILD_LOG, 4096, build_c, NULL );
        printf( "Build Log for %s_program:\n%s\n", "example", build_c );
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
    if(err != CL_SUCCESS) {
        printf("clCreateKernel error %d \n", err);
    }
    // Create the input and output arrays in device memory for our calculation
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) { printf("clCreateBuffer Error %d \n", err); }
 
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &w);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &h);
    err |= clSetKernelArg(kernel, 3, sizeof(long), &n);
    if(err != CL_SUCCESS) {
        printf("clSetKernelArg error %d \n", err);
    }
    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("clEnqueueNDRangeKernel error %d \n", err);
    }
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    
    for(i=0; i<n; i++) {
        sum += h_c[i];
    }

    printf("final result: %ld \n", sum);
 
    // release OpenCL resources
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_c);

    return sum;

}


/// Main program
int main( int argc, char* argv[] )
{
     
    // Size of vectors
    long n = 10000000;

    int w = 6000;
    int h = 4000;

    clock_t begin, end;
    double time_spent_CPU;
    double time_spent_GPU;

    printf("----------------------\n");
    printf("OpenCL Device Infos\n");
    printf("----------------------\n\n");

    showDeviceInfo();

    printf("----------------------\n");
    printf("Mandelbrot\n");
    printf("----------------------\n\n");

    printf("CPU : started...\n");
    begin = clock();
    
    MandelbrotCPU(w,h);
    
    end = clock();
    time_spent_CPU = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed : %f [s] \n\n", time_spent_CPU);

    printf("GPU : started...\n");
    begin = clock();
    
    MandelbrotOpenCL(w,h);
    
    end = clock();
    time_spent_GPU = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed : %f [s] \n\n", time_spent_GPU);

    printf("Gain : %.3f x \n\n", time_spent_CPU*1.00/time_spent_GPU);

    printf("----------------------\n");
    printf("Vector multiplication\n");
    printf("----------------------\n\n");


    printf("CPU : started...\n");
    begin = clock();
    
    VectorCPU(n);
    
    end = clock();
    time_spent_CPU = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed : %f [s] \n", time_spent_CPU);

    printf("GPU : started...\n");
    begin = clock();
    
    VectorOpenCL(n);
    
    end = clock();
    time_spent_GPU = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed : %f [s] \n\n", time_spent_GPU);

    printf("Gain : %.3f x \n\n", time_spent_CPU*1.00/time_spent_GPU);

    return 0;
}