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

    void executeKernel1();      // 使用 CL_MEM_COPY_HOST_PTR
    void executeKernel2();      // 使用 CL_MEM_ALLOC_HOST_PTR
    void executeKernel3();      // 1x4 以M作为主序
    void executeKernel4();      // 1x4 以N作为主序 （相邻线程在B上是连续的
    void executeKernel5();      // 4x4 
    void executeKernel6();      // 4x4 将A每4行的同一列元素排成相邻
    void executeKernel7();      // 4x4 将B每行的16个元素排成一排
    void executeKernel8();      // 4x4 指针换成累加
    
    double test();

    // void testAllocPtr();

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