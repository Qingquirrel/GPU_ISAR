#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <cstdlib>
#include <complex>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <stdio.h>
#include <chrono>
#include <cmath>

#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "cublas_v2.h"
#include "mythrust.h"


#define WIDTH 256
#define HEIGHT 256

__global__ void complexMagnitude(const cufftComplex* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        out[index] = sqrtf(in[index].x * in[index].x + in[index].y * in[index].y);
    }
}

void performFFT(const char* filename) {
    // Read complex data from file
 /*   std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }

    cufftComplex* h_data;
    h_data = (cufftComplex*)malloc(sizeof(cufftComplex) * WIDTH * HEIGHT);
    file.read(reinterpret_cast<char*>(h_data), sizeof(cufftComplex) * WIDTH * HEIGHT);
    file.close();*/

    //read_method
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read data size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t expectedSize = sizeof(cufftComplex) * WIDTH * HEIGHT;
    if (fileSize != expectedSize) {
        std::cerr << "Unexpected file size: " << fileSize << ", expected: " << expectedSize << std::endl;
        file.close();
        return;
    }

    cufftComplex* h_data = new cufftComplex[WIDTH * HEIGHT];
    file.read(reinterpret_cast<char*>(h_data), fileSize);
    file.close();

    // Allocate device memory
    cufftComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * WIDTH * HEIGHT);

    // Copy input data from host to device
    cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    // Create FFT plans
    cufftHandle plan;
    cufftPlan2d(&plan, HEIGHT, WIDTH, CUFFT_C2C);

    // Execute FFT on device
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Execute FFT along columns
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Compute magnitude of complex data
    float* d_magnitude;
    cudaMalloc((void**)&d_magnitude, sizeof(float) * WIDTH * HEIGHT);
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    complexMagnitude << <gridSize, blockSize >> > (d_data, d_magnitude, WIDTH, HEIGHT);

    // Copy magnitude data from device to host
    float* h_magnitude;
    h_magnitude = (float*)malloc(sizeof(float) * WIDTH * HEIGHT);
    cudaMemcpy(h_magnitude, d_magnitude, sizeof(float) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    // Normalize magnitude data if necessary
    // ...

    // Visualize magnitude data as image using OpenCV
  /*  cv::Mat magnitudeImg(HEIGHT, WIDTH, CV_32FC1, h_magnitude);
    cv::normalize(magnitudeImg, magnitudeImg, 0, 255, cv::NORM_MINMAX);
    magnitudeImg.convertTo(magnitudeImg, CV_8UC1);

    cv::imshow("Magnitude Image", magnitudeImg);
    cv::waitKey(0);*/

    cv::Mat originalImg(HEIGHT, WIDTH, CV_32FC2); // 注意使用CV_32FC2，因为数据是复数形式的
    file.read(reinterpret_cast<char*>(originalImg.data), sizeof(float) * 2 * WIDTH * HEIGHT);
    cv::imshow("Original Data", originalImg);
    cv::waitKey(0);


    // Clean up
    free(h_data);
    free(h_magnitude);
    cudaFree(d_data);
    cudaFree(d_magnitude);
    cufftDestroy(plan);
}

int main() {
    performFFT("D:\\AAA_hot_pot\\GPU_image\\Yak42.dat");
    return 0;
}
