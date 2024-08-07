#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <mat.h>

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

#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <mat.h>

#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <mat.h>

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

// CUDA kernel: 执行fftshift操作

__global__ void fftShift(cufftComplex* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfWidth = width / 2;
    int halfHeight = height / 2;

    if (x < width && y < height) {
        int dx = (x + halfWidth) % width;
        int dy = (y + halfHeight) % height;
        int index1 = y * width + x;
        int index2 = dy * width + dx;
        cufftComplex temp = data[index1];
        data[index1] = data[index2];
        data[index2] = temp;
    }
}

// 生成ISAR图像
void generateISAR(const cufftComplex* h_data, int nr, int np, float prf, float dr) {
    // 分配设备内存
    cufftComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * nr * np);

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * nr * np, cudaMemcpyHostToDevice);

    // 创建FFT计划
    cufftHandle plan;
    cufftPlan2d(&plan, nr, np, CUFFT_C2C);

    // 执行FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    dim3 blockSize(16, 16);
    dim3 gridSize((np + blockSize.x - 1) / blockSize.x, (nr + blockSize.y - 1) / blockSize.y);
    fftShift << <gridSize, blockSize >> > (d_data, np, nr);

    // 计算绝对值
    float* d_magnitude;
    cudaMalloc((void**)&d_magnitude, sizeof(float) * nr * np);
    complexMagnitude << <gridSize, blockSize >> > (d_data, d_magnitude, np, nr);

    // 复制数据从设备到主机
    float* h_magnitude;
    h_magnitude = (float*)malloc(sizeof(float) * nr * np);
    cudaMemcpy(h_magnitude, d_magnitude, sizeof(float) * nr * np, cudaMemcpyDeviceToHost);

    // 创建ISAR图像
    cv::Mat ISARImg(nr, np, CV_32FC1, h_magnitude);

    // 计算幅度
    cv::Mat magnitude;
    cv::absdiff(ISARImg, cv::Scalar(0), magnitude);

    // 对数变换
    cv::log(magnitude, magnitude);
    ISARImg = 20 * magnitude;

    double minRange = -prf / 2.0;
    double maxRange = prf / 2.0;
    double minDoppler = -dr * nr / 2.0;
    double maxDoppler = dr * nr / 2.0;

    // 创建窗口并绘制ISAR图像
    cv::Mat canvas(nr, np, CV_8UC3); // 修改为CV_8UC3，因为伪彩色图像需要三个通道
    cv::Mat normalizedISARImg;
    cv::normalize(ISARImg, normalizedISARImg, 0, 255, cv::NORM_MINMAX, CV_8UC1); // 将单通道的ISAR图像归一化为8位灰度图像
    //cv::applyColorMap(normalizedISARImg, canvas, cv::COLORMAP_JET);
    cv::applyColorMap(normalizedISARImg, canvas, cv::COLORMAP_HOT);
    cv::namedWindow("ISAR Image", cv::WINDOW_NORMAL);
    cv::imshow("ISAR Image", canvas);
    cv::Mat canvas_with_axes = canvas.clone();

    // 绘制横轴
    cv::line(canvas_with_axes, cv::Point(0, canvas.rows / 2), cv::Point(canvas.cols, canvas.rows / 2), cv::Scalar(255, 255, 255), 1);

    // 绘制纵轴
    cv::line(canvas_with_axes, cv::Point(canvas.cols / 2, 0), cv::Point(canvas.cols / 2, canvas.rows), cv::Scalar(255, 255, 255), 1);

    // 添加横轴标签
    cv::putText(canvas_with_axes, "Doppler (Hz)", cv::Point(canvas.cols - 80, canvas.rows / 2 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // 添加纵轴标签
    cv::putText(canvas_with_axes, "Range (meter)", cv::Point(canvas.cols / 2 + 10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    cv::namedWindow("ISAR Image with Axes", cv::WINDOW_NORMAL);
    // 显示带有坐标轴的图像
    cv::imshow("ISAR Image with Axes", canvas_with_axes);

    cv::waitKey(0);

    // 释放内存
    free(h_magnitude);
    cudaFree(d_data);
    cudaFree(d_magnitude);
    cufftDestroy(plan);
}

int main() {
    float c = 2.99792458e8;
    float drs = 1.0;
    float bw = 6e9 / drs;
    float Pri = 120e-6;
    float prf = 1 / Pri;

    MATFile* pmat;
    mxArray* pmxArray;
    pmat = matOpen("D:\\AAA_hot_pot\\simple_isar_image\\Yak42\\Data\\Yak42.mat", "r");
    if (pmat == NULL) {
        std::cerr << "Error opening file." << std::endl;
        return -1;
    }

    pmxArray = matGetVariable(pmat, "y");
    if (pmxArray == NULL) {
        std::cerr << "Error reading variable 'y' from file." << std::endl;
        return -1;
    }

    // 检查数组维度
    int nr = mxGetM(pmxArray);
    int np = mxGetN(pmxArray);
    if (mxGetNumberOfDimensions(pmxArray) != 2 || nr != NR || np != NP) {
        std::cerr << "Unexpected file size or dimensions." << std::endl;
        return -1;
    }

    // 获取雷达数据
    cufftComplex* h_data;
    h_data = (cufftComplex*)mxGetData(pmxArray);

    // 执行ISAR成像
    generateISAR(h_data, nr, np, prf, c / (2.0f * bw));

    // 释放内存
    mxDestroyArray(pmxArray);
    matClose(pmat);

    return 0;
}

//data读取
// 文件名
void readRowwiseComplexData(const char* filename, std::vector<cufftComplex>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening data file" << std::endl;
        return;
    }

    for (int i = 0; i < data.size(); ++i) {
        float realPart, imagPart;
        file.read(reinterpret_cast<char*>(&realPart), sizeof(float));
        if (!file) {
            std::cerr << "Error reading real part from file" << std::endl;
            return;
        }
        data[i].x = realPart;
    }

    for (int i = 0; i < data.size(); ++i) {
        float imagPart;
        file.read(reinterpret_cast<char*>(&imagPart), sizeof(float));
        if (!file) {
            std::cerr << "Error reading imaginary part from file" << std::endl;
            return;
        }
        data[i].y = imagPart;
    }
}
void readRowwiseComplexData2(const char* filename, std::vector<cufftComplex>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening data file" << std::endl;
        return;
    }

    // 为读取一行的数据分配临时缓冲区
    std::vector<float> rowBuffer(2 * NP);

    for (int i = 0; i < data.size(); ++i) {
        // 读取一行的实部和虚部
        file.read(reinterpret_cast<char*>(rowBuffer.data()), sizeof(float) * 2 * NP);
        if (!file) {
            std::cerr << "Error reading data from file" << std::endl;
            return;
        }

        // 将实部和虚部合成为复数并存储
        for (int j = 0; j < NP; ++j) {
            data[i * NP + j].x = rowBuffer[2 * j];
            data[i * NP + j].y = rowBuffer[2 * j + 1];
        }
    }
}

    const char* filename = "D:\\AAA_hot_pot\\GPU_image\\yak42_ISAR\\Yak42_transposed.mat";

    // 分配内存用于存储数据
    std::vector<cufftComplex> data(NR * NP);

    // 读取数据并将其存储在数组中
    readRowwiseComplexData2(filename, data);

        //// 计算最大绝对值
    //float maxAbsValue = 0.0f;
    //const cufftComplex* maxElementPtr = nullptr;
    //for (int i = 0; i < NR * NP; ++i) {
    //    float val = sqrtf(h_result[i].x * h_result[i].x + h_result[i].y * h_result[i].y);
    //    maxAbsValue = std::max(maxAbsValue, val);
    //    if (val == maxAbsValue) {
    //        maxElementPtr = &h_result[i];
    //    }
    //}

    //std::cout << "Max Absolute Value: " << maxAbsValue << std::endl;


    //// 将最大绝对值拷贝到GPU内存
    //float* d_maxAbsValue;
    //cudaMalloc((void**)&d_maxAbsValue, sizeof(float));
    //cudaMemcpy(d_maxAbsValue, &maxAbsValue, sizeof(float), cudaMemcpyHostToDevice);
