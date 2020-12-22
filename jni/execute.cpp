#include "compute_opencl.h"

int main(){
    
    std::cout << "hello OCL matrix multiplation !" << std::endl;

    cl_uint numPlatforms;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    
    std::cout << "success load openCL.so" << std::endl;
}