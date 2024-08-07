#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <mat.h>

#include <cuda_runtime.h>
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

#define NR 256 // 雷达数据的行数
#define NP 256 // 雷达数据的列数

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// CUDA kernel: 计算复数数据的绝对值
__global__ void complexMagnitude(const cufftComplex* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        out[index] = sqrtf(in[index].x * in[index].x + in[index].y * in[index].y);
    }
}

// CUDA kernel: 运动补偿后的二维FFT
__global__ void motionCompensationFFT(const cufftComplex* in, cufftComplex* out, int width, int height, float prf, float dr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        // 计算频率
        float f = (x - width / 2) * prf / width;
        // 计算范围
        float r = (y - height / 2) * dr;
        // 计算相位
        float phase = -2 * M_PI * f * r;
        float cosPhase = cos(phase);
        float sinPhase = sin(phase);

        // 进行频率相位调整
        out[index].x = in[index].x * cosPhase - in[index].y * sinPhase;
        out[index].y = in[index].x * sinPhase + in[index].y * cosPhase;
    }
}

void exportISARImage(const std::string& filename, float* d_data, int width, int height) {
    // 分配主机内存
    float* h_data = (float*)malloc(sizeof(float) * width * height);

    // 将数据从设备复制到主机
    cudaMemcpy(h_data, d_data, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // 创建OpenCV图像对象
    cv::Mat ISARImg(height, width, CV_32FC1, h_data);

    // 将图像保存到文件
    cv::imwrite(filename, ISARImg);

    // 释放内存
    free(h_data);
}

void generateISAR(const cufftComplex* h_data, float prf, float dr) {
    // 分配设备内存
    cufftComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * NR * NP);

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * NR * NP, cudaMemcpyHostToDevice);

    // 创建FFT计划
    cufftHandle plan;
    cufftPlan2d(&plan, NR, NP, CUFFT_C2C);

    // 执行FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // 执行FFT沿着列
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // 计算绝对值
    float* d_magnitude;
    cudaMalloc((void**)&d_magnitude, sizeof(float) * NR * NP);
    dim3 blockSize(16, 16);
    dim3 gridSize((NP + blockSize.x - 1) / blockSize.x, (NR + blockSize.y - 1) / blockSize.y);
    complexMagnitude << <gridSize, blockSize >> > (d_data, d_magnitude, NP, NR);

    // 复制数据从设备到主机
    float* h_magnitude;
    h_magnitude = (float*)malloc(sizeof(float) * NR * NP);
    cudaMemcpy(h_magnitude, d_magnitude, sizeof(float) * NR * NP, cudaMemcpyDeviceToHost);

    // 规范化和对数变换
    //cv::Mat ISARImg(NR, NP, CV_32FC1, h_magnitude);
    //float maxVal = 0.0f;
    //for (int i = 0; i < ISARImg.rows; ++i) {
    //    for (int j = 0; j < ISARImg.cols; ++j) {
    //        float val = ISARImg.at<float>(i, j);
    //        if (val > maxVal) {
    //            maxVal = val;
    //        }
    //    }
    //}

    //// 通过最大值进行规范化
    //for (int i = 0; i < ISARImg.rows; ++i) {
    //    for (int j = 0; j < ISARImg.cols; ++j) {
    //        ISARImg.at<float>(i, j) /= maxVal;
    //    }
    //}
    //cv::log(ISARImg, ISARImg); // 对数变换

    cv::Mat ISARImg(NR, NP, CV_32FC1, h_magnitude);
    double minVal, maxVal;
    cv::minMaxLoc(ISARImg, &minVal, &maxVal);
    ISARImg = (ISARImg - minVal) / (maxVal - minVal);
    cv::log(ISARImg, ISARImg);

    exportISARImage("ISAR_Image.png", d_magnitude, NP, NR);


    // 可视化图像
    cv::imshow("ISAR Image", ISARImg);
    cv::waitKey(0);

    // 释放内存
    free(h_magnitude);
    cudaFree(d_data);
    cudaFree(d_magnitude);
    cufftDestroy(plan);
}


int main() {
    float c = 2.99792458e8f;   // 光速
    float drs = 1.0f;          // 距离分辨率因子
    float bw = 6e9f / drs;     // 带宽
    float Pri = 120e-6f;      // 脉冲重复间隔

    float prf = 1.0f / Pri;    // 脉冲重复频率
    float dr = c / (2.0f * bw); // 距离分辨率

    MATFile* pmat;
    mxArray* pmxArray;
    pmat = matOpen("D:\\AAA_hot_pot\\simple_isar_image\\Yak42\\Data\\Yak42.mat", "r"); // 替换成您的.mat文件路径
    if (pmat == NULL) {
        std::cerr << "Error opening file." << std::endl;
        return -1;
    }

    pmxArray = matGetVariable(pmat, "y"); // 假设.mat文件中的变量名为 "y"
    if (pmxArray == NULL) {
        std::cerr << "Error reading variable 'y' from file." << std::endl;
        return -1;
    }

    // 检查数组维度
    if (mxGetNumberOfDimensions(pmxArray) != 2 || mxGetM(pmxArray) != NR || mxGetN(pmxArray) != NP) {
        std::cerr << "Unexpected file size or dimensions." << std::endl;
        return -1;
    }

    // 获取雷达数据
    cufftComplex* h_data;
    h_data = (cufftComplex*)mxGetData(pmxArray);

    // 执行ISAR成像
    generateISAR(h_data, prf, dr);
    

    // 释放内存
    mxDestroyArray(pmxArray);
    matClose(pmat);

    return 0;

}
