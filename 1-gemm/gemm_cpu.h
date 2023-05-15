#pragma once

void gemm_cpu(float* C, const float* A, const float* B, int hA, int wA, int wB) {
  for (int i = 0; i < hA; i++) {
    for (int j = 0; j < wB; j++) {
      float sum = 0.0;

      for (int k = 0; k < wA; k++) {
        sum += A[i * wA + k] * B[k * wB + j];
      }

      C[i * wB + j] = sum;
    }
  }
}