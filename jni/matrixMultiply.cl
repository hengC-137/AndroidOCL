

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
        // acc += (A[globalRow * K + k] * B[k * N + globalCol]);
    }
    C[globalRow * K + globalCol] = acc;
}