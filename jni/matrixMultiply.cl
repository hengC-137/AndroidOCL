

__kernel void myGEMM(const int M, const int N, const int K, const __global float* A,
                     const __global float* B, __global float* C) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    float acc = 0.0f;
    for (int k = 0; k < K; k++){
        
        // printf("K = %d \n", k);
        // printf("A = %f, B = %f \n", A[globalRow * K + k], B[k * N + globalCol]);
        // float tempA = A[globalRow * K + k];
        // printf("tempA = %f \n", tempA);
        // float tempB = B[k * N + globalCol];
        // printf("tempB = %f \n", tempB);
        // float temp = A[globalRow * K + k] * B[k * N + globalCol];
        // printf("temp = %f \n", temp);
        // acc += temp;
        // printf("acc = %f \n", acc);
        acc += (A[globalRow * K + k] * B[k * N + globalCol]);
    }
    C[globalRow * K + globalCol] = acc;
}

__kernel void myGEMM2(const int M, const int N, const int K, const __global float* A,
                     const __global float* B, __global float* C) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float ar = 0.0f;
    float4 bv = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k++){
        
    //     // printf("K = %d \n", k);
    //     // printf("A = %f, B = %f \n", A[globalRow * K + k], B[k * N + globalCol]);
    //     // float tempA = A[globalRow * K + k];
    //     // printf("tempA = %f \n", tempA);
    //     // float tempB = B[k * N + globalCol];
    //     // printf("tempB = %f \n", tempB);
    //     // float temp = A[globalRow * K + k] * B[k * N + globalCol];
    //     // printf("temp = %f \n", temp);
    //     // acc += temp;
    //     // printf("acc = %f \n", acc);

        ar = A[globalRow * K + k];
        bv = vload4(0, &B[k * N + globalCol * 4]);
        // printf("row = %d col = %d   b x= %f \n", globalRow, globalCol, bv.x);
        // printf("row = %d col = %d   b y= %f \n", globalRow, globalCol, bv.y);
        // printf("row = %d col = %d   b z= %f \n", globalRow, globalCol, bv.z);
        // printf("row = %d col = %d   b w= %f \n", globalRow, globalCol, bv.w);

        acc.x += ar * bv.x;
        acc.y += ar * bv.y;
        acc.z += ar * bv.z;
        acc.w += ar * bv.w;
    }
    C[globalRow * K + globalCol * 4] = acc.x;
    C[globalRow * K + globalCol * 4 + 1] = acc.y;
    C[globalRow * K + globalCol * 4 + 2] = acc.z;
    C[globalRow * K + globalCol * 4 + 3] = acc.w;
}

__kernel void myGEMM3(const int M, const int N, const int K, const __global float* A,
                     const __global float* B, __global float* C) {
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(0);

   float ar = 0.0f;
    float4 bv = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k++){
        
    //     // printf("K = %d \n", k);
    //     // printf("A = %f, B = %f \n", A[globalRow * K + k], B[k * N + globalCol]);
    //     // float tempA = A[globalRow * K + k];
    //     // printf("tempA = %f \n", tempA);
    //     // float tempB = B[k * N + globalCol];
    //     // printf("tempB = %f \n", tempB);
    //     // float temp = A[globalRow * K + k] * B[k * N + globalCol];
    //     // printf("temp = %f \n", temp);
    //     // acc += temp;
    //     // printf("acc = %f \n", acc);

        ar = A[globalRow * K + k];
        bv = vload4(0, &B[k * N + globalCol * 4]);
        // printf("row = %d col = %d   b x= %f \n", globalRow, globalCol, bv.x);
        // printf("row = %d col = %d   b y= %f \n", globalRow, globalCol, bv.y);
        // printf("row = %d col = %d   b z= %f \n", globalRow, globalCol, bv.z);
        // printf("row = %d col = %d   b w= %f \n", globalRow, globalCol, bv.w);

        acc.x += ar * bv.x;
        acc.y += ar * bv.y;
        acc.z += ar * bv.z;
        acc.w += ar * bv.w;
    }
    C[globalRow * K + globalCol * 4] = acc.x;
    C[globalRow * K + globalCol * 4 + 1] = acc.y;
    C[globalRow * K + globalCol * 4 + 2] = acc.z;
    C[globalRow * K + globalCol * 4 + 3] = acc.w;
}

__kernel void myGEMM4(const int M, const int N, const int K, const __global float* A,
                     const __global float* B, __global float* C) {
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(0);

   float ar = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 bv = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k++){
        
    //     // printf("K = %d \n", k);
    //     // printf("A = %f, B = %f \n", A[globalRow * K + k], B[k * N + globalCol]);
    //     // float tempA = A[globalRow * K + k];
    //     // printf("tempA = %f \n", tempA);
    //     // float tempB = B[k * N + globalCol];
    //     // printf("tempB = %f \n", tempB);
    //     // float temp = A[globalRow * K + k] * B[k * N + globalCol];
    //     // printf("temp = %f \n", temp);
    //     // acc += temp;
    //     // printf("acc = %f \n", acc);

        ar = A[globalRow * K + k];
        bv = vload4(0, &B[k * N + globalCol * 4]);
        // printf("row = %d col = %d   b x= %f \n", globalRow, globalCol, bv.x);
        // printf("row = %d col = %d   b y= %f \n", globalRow, globalCol, bv.y);
        // printf("row = %d col = %d   b z= %f \n", globalRow, globalCol, bv.z);
        // printf("row = %d col = %d   b w= %f \n", globalRow, globalCol, bv.w);

        acc.x += ar * bv.x;
        acc.y += ar * bv.y;
        acc.z += ar * bv.z;
        acc.w += ar * bv.w;
    }
    C[globalRow * K + globalCol * 4] = acc.x;
    C[globalRow * K + globalCol * 4 + 1] = acc.y;
    C[globalRow * K + globalCol * 4 + 2] = acc.z;
    C[globalRow * K + globalCol * 4 + 3] = acc.w;
}