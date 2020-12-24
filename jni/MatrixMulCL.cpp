#include "MatrixMulCL.h"

void initMat(float* mat, size_t matSize, int mode){
    for (size_t i = 0; i < matSize; i++){
        switch (mode)
        {
        case 0:
            mat[i] = 0;
            break;
        case 1:
            mat[i] = 0.1f;
            break;
        case 2: 
            mat[i] = ((rand() / float(RAND_MAX)) - 0.5) * 10;
            // mat[i] = i;
            break;
        default:
            break;
        }
    }
}

float myAbs(float x) {
    return x > 0 ? x : -x;
}

void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}


MatrixMulCL::MatrixMulCL(/* args */){}

MatrixMulCL::~MatrixMulCL(){}


cl_int MatrixMulCL::getPlatform(){

    cl_uint numPlatforms;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS){
        std::cout << "Error: Getting Platforms failed!" << std::endl;
        return status;
    }

    std::cout << numPlatforms << " platform available." << std::endl;

    if (numPlatforms > 0){
        cl_platform_id* platforms = (cl_platform_id* ) malloc (numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform = platforms[0];
        free(platforms);
        return CL_SUCCESS;
    }
    else{
        return status;
    }
}

cl_int MatrixMulCL::getCLDeviceId() {
    cl_uint numDevices = 0;
    cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    std::cout << numDevices << " gpu device available" << std::endl;
    if (status != CL_SUCCESS){
        return status;
    }

    if (numDevices > 0) {
        devices = (cl_device_id*) malloc (numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        return status;
    }
    return -1;
}

int MatrixMulCL::convertToString(const char *filename, std::string& s){
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
    if (f.is_open()){
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if (!str) {
            f.close();
            return 0;
        }
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout << "Error: failed to open file\n" << filename << std::endl;
    return -1;
}

void MatrixMulCL::initCL(){
    status = getPlatform();
    if(status != CL_SUCCESS){
        std::cout << "get platform faild, status : " << status << std::endl;
        return;
    }

    cl_context_properties context_properties[] =
    { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
    };
    context_properties[1] = (cl_context_properties)platform;

    status = getCLDeviceId();    
    if (status != CL_SUCCESS){
        std::cout << "Error: Getting Devices failed!" << std::endl;
        return;
    }

    // context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);
    context = clCreateContext(context_properties, 1, devices, NULL, NULL, NULL);
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

    std::string sourceStr;
    status = convertToString(filename, sourceStr);
    
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);
    if (status != CL_SUCCESS) {
        std::cout << "Create program failed" << std::endl;
    }

    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if (status != CL_SUCCESS){
        std::cout << "Error: Build Program failed! error : " << status << std::endl;
        return;
    }
    std::cout << "openCL init finish" << std::endl;
    
}

void MatrixMulCL::initMatrix(int M_, int N_, int K_){
    M = M_;
    N = N_;
    K = K_;

    A = (float*) malloc (sizeof(float) * (M *K));
    B = (float*) malloc (sizeof(float) * (N *K));
    C = (float*) malloc (sizeof(float) * (K *K));
    C_gold = (float*) malloc (sizeof(float) * (K *K));
    
    initMat(A, M*K, 2);
    initMat(B, N*K, 2);
    initMat(C, K*K, 0);
    initMat(C_gold, M*N, 0);

    // std::cout << "A :" << std::endl;
    // for (size_t i = 0; i < M; i++)
    // {
    //     for (size_t j = 0; j < K; j++)
    //     {
    //         std::cout << A[i * K + j] << "  ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "B :" << std::endl;
    // for (size_t i = 0; i < K; i++)
    // {
    //     for (size_t j = 0; j < N; j++)
    //     {
    //         std::cout << B[i * K + j] << "  ";
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << "finish init matrix" << std::endl;
}


double MatrixMulCL::executeKernel(){
    struct timespec start, end;
    
    kernel = clCreateKernel(program, "myGEMM", &status);
    if (status != CL_SUCCESS)
    {
        std::cout << "Error: Create kernel failed! error code : " << status << std::endl;
    }
    Amem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (M * K) * sizeof(float), A, &status);
    Bmem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (K * N) * sizeof(float), B, &status);
    Cmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (K * K) * sizeof(float), NULL, &status);
    
    size_t globalRange[2] = {(size_t)M, (size_t)N};

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    status = clSetKernelArg(kernel, 0, sizeof(int), (void*)&M);
    status |= clSetKernelArg(kernel, 1, sizeof(int), (void*)&N);
    status |= clSetKernelArg(kernel, 2, sizeof(int), (void*)&K);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&Amem);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&Bmem);
    status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&Cmem);

    cl_event eventPoint;
    status |= clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalRange, NULL, 0, NULL, &eventPoint);
    clWaitForEvents(1, &eventPoint);
    clReleaseEvent(eventPoint);
    status |= clEnqueueReadBuffer(commandQueue, Cmem, CL_TRUE, 0, K * K * sizeof(float), C, 0, NULL, NULL);


    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    
    status = clReleaseMemObject(Amem);
    status = clReleaseMemObject(Bmem);
    status = clReleaseMemObject(Cmem);
    return get_time(&start, &end);
    // return 0;

}

double MatrixMulCL::nativeMatrixMul(){

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (size_t m = 0; m < M; m++){
        // std::cout << "M = " << m << std::endl;
        for (size_t n = 0; n < N; n++){
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++){
                acc += A[m * K + k] * B[N * k + n];
            }
            C_gold[m * N + n] = acc;   
        }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    return get_time(&start, &end);
}

void MatrixMulCL::campareResult(){
    for (size_t i = 0; i < K * K; i++)
    {
        if (myAbs(C[i] - C_gold[i]) > 0.01){
            std::cout << i << "  " << C[i] << "  " << C_gold[i] << std::endl;
        }
    }
    
}

void MatrixMulCL::freeMatrix(){
    if (A != nullptr) {
        free(A);
        A = nullptr;
    }
    if (B != nullptr) {
        free(B);
        B = nullptr;
    }
    if (C != nullptr) {
        free(C);
        C = nullptr;
    }
    if (C_gold != nullptr) {
        free(C_gold);
        C_gold = nullptr;
    }
}

void MatrixMulCL::releaseMatrixMulCL(){
    status = clReleaseKernel(kernel);
    status = clReleaseProgram(program);
    status = clReleaseCommandQueue(commandQueue);
    status = clReleaseContext(context);
}