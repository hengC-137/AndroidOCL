#include "compute_opencl.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>

const int LENTH = 4;

class MatrixMulCL
{
public:
    MatrixMulCL();
    ~MatrixMulCL();

    void initCL();
    void initMatrix(int, int, int);
    double nativeMatrixMul(); 
    double executeKernel(); 
    void prepareMem(); 
    void campareResult();
    void releaseMatrixMulCL();
    void freeMatrix();


private:
    cl_int status;
    cl_platform_id platform;
    cl_device_id* devices = nullptr;
    cl_context context;
    cl_program program;
    cl_command_queue commandQueue;
    cl_kernel kernel;

    cl_mem Amem; 
    cl_mem Bmem; 
    cl_mem Cmem;
    
    cl_int getPlatform();
    cl_int getCLDeviceId();
    int convertToString(const char *filename, std::string& s);

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* C_gold = nullptr;

    const char *filename = "/data/data/test/matrixMultiply.cl";

    int M = 0;
    int N = 0;
    int K = 0;
};

static double gtod_ref_time_sec = 0.0;

static double get_time(struct timespec *start, struct timespec *end) {
    return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * 1e-9;
}