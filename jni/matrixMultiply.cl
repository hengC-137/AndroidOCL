

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

    float ar1 = 0.0f;
    float ar2 = 0.0f;
    float ar3 = 0.0f;
    float ar0 = 0.0f;
    float4 bv = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k++){
        ar0 = A[(globalRow * 4) * K + k];
        ar1 = A[(globalRow * 4 + 1) * K + k];
        ar2 = A[(globalRow * 4 + 2) * K + k];
        ar3 = A[(globalRow * 4 + 3) * K + k];

        bv = vload4(0, &B[k * N + globalCol * 4]);

        acc0.x += ar0 * bv.x;
        acc0.y += ar0 * bv.y;
        acc0.z += ar0 * bv.z;
        acc0.w += ar0 * bv.w;
        
        acc1.x += ar1 * bv.x;
        acc1.y += ar1 * bv.y;
        acc1.z += ar1 * bv.z;
        acc1.w += ar1 * bv.w;
        
        acc2.x += ar2 * bv.x;
        acc2.y += ar2 * bv.y;
        acc2.z += ar2 * bv.z;
        acc2.w += ar2 * bv.w;
        
        acc3.x += ar3 * bv.x;
        acc3.y += ar3 * bv.y;
        acc3.z += ar3 * bv.z;
        acc3.w += ar3 * bv.w;
    }

    vstore4(acc0, 0, &C[globalRow * N * 4 + globalCol * 4]);
    vstore4(acc1, 0, &C[(globalRow * 4 + 1) * N + globalCol * 4]);
    vstore4(acc2, 0, &C[(globalRow * 4 + 2) * N + globalCol * 4]);
    vstore4(acc3, 0, &C[(globalRow * 4 + 3) * N + globalCol * 4]);
}

__kernel void myGEMM5(const int M, const int N, const int K, const __global float* A,
                      const __global float* B, __global float* C) {
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(0);

    float4 av = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 bv = {0.0f, 0.0f, 0.0f, 0.0f};

    float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k++){
        av = vload4(0, &A[globalRow * 4 * K + 4 * k]);
        bv = vload4(0, &B[k * N + globalCol * 4]);

        acc0.x += av.x * bv.x;
        acc0.y += av.x * bv.y;
        acc0.z += av.x * bv.z;
        acc0.w += av.x * bv.w;
        
        acc1.x += av.y * bv.x;
        acc1.y += av.y * bv.y;
        acc1.z += av.y * bv.z;
        acc1.w += av.y * bv.w;
        
        acc2.x += av.z * bv.x;
        acc2.y += av.z * bv.y;
        acc2.z += av.z * bv.z;
        acc2.w += av.z * bv.w;
        
        acc3.x += av.w * bv.x;
        acc3.y += av.w * bv.y;
        acc3.z += av.w * bv.z;
        acc3.w += av.w * bv.w;
    }

    vstore4(acc0, 0, &C[globalRow * N * 4 + globalCol * 4]);
    vstore4(acc1, 0, &C[(globalRow * 4 + 1) * N + globalCol * 4]);
    vstore4(acc2, 0, &C[(globalRow * 4 + 2) * N + globalCol * 4]);
    vstore4(acc3, 0, &C[(globalRow * 4 + 3) * N + globalCol * 4]);
}

__kernel void myGEMM6(const int M, const int N, const int K, const __global float* A,
                      const __global float* B, __global float* C) {
    const int globalRow = get_global_id(1);
    const int globalCol = get_global_id(0);

    float4 av = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 bv = {0.0f, 0.0f, 0.0f, 0.0f};

    float4 acc0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 acc3 = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k++){
        av = vload4(0, &A[globalRow * 4 * K + 4 * k]);
        bv = vload4(0, &B[k * N + globalCol * 4]);

        acc0.x += av.x * bv.x;
        acc0.y += av.x * bv.y;
        acc0.z += av.x * bv.z;
        acc0.w += av.x * bv.w;
        
        acc1.x += av.y * bv.x;
        acc1.y += av.y * bv.y;
        acc1.z += av.y * bv.z;
        acc1.w += av.y * bv.w;
        
        acc2.x += av.z * bv.x;
        acc2.y += av.z * bv.y;
        acc2.z += av.z * bv.z;
        acc2.w += av.z * bv.w;
        
        acc3.x += av.w * bv.x;
        acc3.y += av.w * bv.y;
        acc3.z += av.w * bv.z;
        acc3.w += av.w * bv.w;
    }

    vstore4(acc0, 0, &C[globalRow * N * 4 + globalCol * 4]);
    vstore4(acc1, 0, &C[(globalRow * 4 + 1) * N + globalCol * 4]);
    vstore4(acc2, 0, &C[(globalRow * 4 + 2) * N + globalCol * 4]);
    vstore4(acc3, 0, &C[(globalRow * 4 + 3) * N + globalCol * 4]);
}

