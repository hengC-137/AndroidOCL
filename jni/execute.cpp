#include "compute_opencl.h"
#include "MatrixMulCL.h"
#include <thread>

void func(){
    MatrixMulCL* mmcl = new MatrixMulCL();
    mmcl->initCL();

    double onceTime = 0, minTime;

    for (size_t lenth = 128; lenth < 4096; lenth += 64)
    {
        minTime = SIZE_MAX;
        mmcl->initMatrix(lenth, lenth, lenth);
        mmcl->prepareMem();
        for (size_t i = 0; i < 10; i++)
        {
            mmcl->executeKernel2();
        }
        mmcl->freeMatrix();
        printf("lenth = %d  best cost time %lf \n", lenth, minTime);
    }

    mmcl->releaseMatrixMulCL();
    
}


int main(){

    // std::thread t(func);
    // t.join();

    MatrixMulCL* mmcl = new MatrixMulCL();
    mmcl->initCL();
    // mmcl->nativeMatrixMul();


    // double onceTime = 0, minTime;

    // for (size_t lenth = 128; lenth < 2049; lenth += 64)
    // {
    //     minTime = SIZE_MAX;
    //     mmcl->initMatrix(lenth, lenth, lenth);
    //     for (size_t i = 0; i < 10; i++)
    //     {
    //         onceTime = mmcl->test();
    //         minTime = onceTime < minTime ? onceTime : minTime;
    //     }
    //     mmcl->freeMatrix();
    //     std::cout << "lenth = " << lenth << " best cost time " << minTime << std::endl;
    // }

    // mmcl->executeKernel2();
    // mmcl->executeKernel3();
    // mmcl->executeKernel4();
    // mmcl->executeKernel5();
    // mmcl->executeKernel6();
    // mmcl->executeKernel7();
    // mmcl->executeKernel8();
    
 


    // double t = mmcl->executeKernel();
    // std::cout << " t= " << t << std::endl;
    // mmcl->campareResult();
    // mmcl->releaseMatrixMulCL();
    
 

    
    // std::cout << "hello OCL matrix multiplation !" << std::endl;

    // cl_uint numPlatforms;
    // cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    
    // std::cout << "success load openCL.so" << std::endl;
}