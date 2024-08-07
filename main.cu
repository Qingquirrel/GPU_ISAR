//#include "yak.h"
//#include "mythrust.h"
//#include <vector>
#include <matio.h>
#include "Envelope.h"

typedef thrust::complex<float> comThr;


#define NR 256 // 雷达数据的行数
#define NP 256 // 雷达数据的列数





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

void fftshiftThrust(cuComplex* d_data, int data_length)
{
    comThr* thr_temp_d_data = reinterpret_cast<comThr*>(d_data);
    thrust::device_ptr<comThr>thr_d_data = thrust::device_pointer_cast(thr_temp_d_data);
    thrust::swap_ranges(thrust::device, thr_d_data, thr_d_data + data_length / 2, thr_d_data + data_length / 2);
}


// CUDA kernel: 归一化复数数据
__global__ void normalize1(const cufftComplex* in, int maxVal, int size, cufftComplex* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        out[index].x = in[index].x / maxVal;
        out[index].y = in[index].y / maxVal;
    }
}
__global__ void applyLog(float* data, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        data[index] = 20.0f * log10f(data[index]);
    }
}

void normalize(const cufftComplex* in, cufftComplex* out, int maxVal, int size) {
    int numBlocks = (size + 255) / 256;
    normalize1 << <numBlocks, 256 >> > (in, maxVal, size, out);
    cudaDeviceSynchronize();
}

__global__ void complexMagnitude(cufftComplex* in, float* out, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = sqrtf(in[index].x * in[index].x + in[index].y * in[index].y);
    }
}
//包络对齐出图
//__global__ void computeEnvelope(const cufftComplex* input, float* output, int width, int height) {
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (x < width && y < height) {
//        int idx = y * width + x;
//        float real = input[idx].x;
//        float imag = input[idx].y;
//        output[idx] = sqrtf(real * real + imag * imag);
//    }
//}

void computeMagnitude(cufftComplex* in, float* out, int size) {
    int numBlocks = (size + 255) / 256;
    complexMagnitude << <numBlocks, 256 >> > (in, out, size);
    cudaDeviceSynchronize();
}
void initializeParams(int N, thrust::device_vector<float>& d_hamming, thrust::device_vector<thrust::complex<float>>& d_Ps, thrust::device_vector<float>& d_Vec_N) {
    std::vector<float> h_hamming(N);
    std::vector<thrust::complex<float>> h_Ps(N);
    std::vector<float> h_Vec_N(N);

    // 计算 hamming 窗和 Ps
    for (int i = 0; i < N; ++i) {
        h_hamming[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (N - 1));
        h_Ps[i] = h_hamming[i] * thrust::exp(thrust::complex<float>(0, -M_PI * i));
        h_Vec_N[i] = i;
    }

    // 传输到 GPU
    d_hamming = h_hamming;
    d_Ps = h_Ps;
    d_Vec_N = h_Vec_N;
}



//读取实数
void saveMatToFile(const char* filename, const float* data, int nr, int np) {
    MATFile* pmat;
    mxArray* pa;
    pmat = matOpen(filename, "w");
    if (pmat == NULL) {
        std::cerr << "Error creating file " << filename << std::endl;
        return;
    }

    mwSize dims[2] = { nr, np };
    pa = mxCreateNumericMatrix(nr, np, mxSINGLE_CLASS, mxREAL); // 修改数据类型为 mxREAL
    if (pa == NULL) {
        std::cerr << "Error creating mxArray" << std::endl;
        matClose(pmat);
        return;
    }

    // 将实数数据拷贝到 mxArray 中
    float* pData = reinterpret_cast<float*>(mxGetData(pa));
    std::memcpy(pData, data, sizeof(float) * nr * np);

    matPutVariable(pmat, "data", pa);
    matClose(pmat);
}
//读取复数
void saveMatToFile2(const char* filename, const cufftComplex* data, int nr, int np) {
    MATFile* pmat;
    mxArray* pa;
    pmat = matOpen(filename, "w");
    if (pmat == NULL) {
        std::cerr << "Error creating file " << filename << std::endl;
        return;
    }

    mwSize dims[2] = { nr, np };
    pa = mxCreateNumericMatrix(nr, np, mxSINGLE_CLASS, mxCOMPLEX);
    if (pa == NULL) {
        std::cerr << "Error creating mxArray" << std::endl;
        matClose(pmat);
        return;
    }

    // 将复数数据拷贝到 mxArray 中
    cufftComplex* pComplex = reinterpret_cast<cufftComplex*>(mxGetData(pa));
    for (int i = 0; i < nr * np; ++i) {
        pComplex[i].x = data[i].x;
        pComplex[i].y = data[i].y;
    }

    matPutVariable(pmat, "data", pa);
    matClose(pmat);
}
//读取cv数据
void saveMatToFile3(const char* filename, const cv::Mat& mat) {
    // 打开 MAT 文件
    MATFile* pmat = matOpen(filename, "w");
    if (pmat == NULL) {
        std::cerr << "Error creating file " << filename << std::endl;
        return;
    }

    // 创建 mxArray
    mxArray* pa = mxCreateNumericMatrix(mat.rows, mat.cols, mxSINGLE_CLASS, mxREAL);
    if (pa == NULL) {
        std::cerr << "Error creating mxArray" << std::endl;
        matClose(pmat);
        return;
    }

    // 将数据拷贝到 mxArray
    float* pData = reinterpret_cast<float*>(mxGetData(pa));
    if (pData == NULL) {
        std::cerr << "Error getting data pointer" << std::endl;
        mxDestroyArray(pa);
        matClose(pmat);
        return;
    }

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            pData[i * mat.cols + j] = mat.at<float>(i, j);
        }
    }

    // 将 mxArray 写入 MAT 文件
    matPutVariable(pmat, "data", pa);
    matClose(pmat);
}
//逐行读取
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

// 回调函数，用于处理滑动条变化
void on_trackbar(int value, void* userdata)
{
    cv::Mat* img = static_cast<cv::Mat*>(userdata);

    double scale = value / 10.0;
    cv::Mat resized_img;
    cv::resize(*img, resized_img, cv::Size(), scale, scale, cv::INTER_LINEAR);

    // Apply a color map to the image to create a heatmap
    cv::Mat heatmap_img;
    cv::applyColorMap(resized_img, heatmap_img, cv::COLORMAP_JET);

    cv::imshow("Yak42_ISAR", heatmap_img);
}

// 转置复数矩阵
void transposeComplexMatrix(const std::vector<cufftComplex>& input, std::vector<cufftComplex>& output, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            output[j * height + i] = input[i * width + j];
        }
    }
}


int main() {
    float c = 2.99792458e8;
    float drs = 1.0;
    float bw = 6e9 / drs;
    float Pri = 120e-6;
    float prf = 1 / Pri;
    float dr = c / (2 * bw);

    const char* filename = "D:\\AAA_hot_pot\\GPU_image\\yak42_ISAR\\Yak42_transposed.dat";

    // 分配内存用于存储数据
    std::vector<cufftComplex> data(NR * NP);
    std::vector<cufftComplex> data_zhuanzhi(NR * NP);

    // 读取数据并将其存储在数组中
    readRowwiseComplexData(filename, data);

    // Create an instance of EnvelopeProcessor
    EnvelopeProcessor Processor(NR, NP);

    // Compute the envelope and display the result
    cv::Mat envelope_image;
    cv::Mat envelope2_image;
    cv::Mat envelope3_image;
   // Processor.Envelope_show(data, envelope_image);
    Processor.Envelope_show2(data);
    //Processor.computeAndShowEveryNthRowEnvelope(data,25);
      // 转置矩阵
    // 转置后的数据
    std::vector<cufftComplex> transposedInput(NR * NP);
    transposeComplexMatrix(data, transposedInput, NR, NP);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始事件
    cudaEventRecord(start, 0);

    Processor.processAndAlign(transposedInput, envelope2_image);

    // 记录结束事件
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // 计算时间差
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // 打印执行时间
    std::cout << "processAndAlign 执行时间: " << elapsedTime << " 毫秒" << std::endl;
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<cufftComplex> juliduiqi_data(NR * NP);
    transposeComplexMatrix(transposedInput, juliduiqi_data, NR, NP);

    Processor.Envelope_show(juliduiqi_data, envelope3_image);

    // 根据 MATLAB 的图像生成逻辑生成图像
    for (int l = 0; l < NP; l += 50) {
        cv::Mat temp;
        cv::normalize(envelope2_image.row(l), temp, 0, 255, cv::NORM_MINMAX);
        cv::line(envelope2_image, cv::Point(0, l), cv::Point(NR - 1, l), cv::Scalar(255, 0, 0), 2);
        envelope2_image.row(l) = envelope2_image.row(l) + l;
    }

    cv::imshow("Aligned Data", envelope2_image);
    cv::waitKey(0);
    ////////////////////////////////////////////////




    // 打印前几个数据项
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_data[" << i << "] = " << data[i].x << " + " << data[i].y << "i" << std::endl;
    }

    const char* filename1 = "Copy_of_Yak42_dat.mat";
    saveMatToFile2(filename1, data.data(), NR, NP);

    cudaDeviceReset(); // 重置CUDA设备状态
    cudaSetDevice(0); // 设置CUDA设备为0号设备
    cudaDeviceSynchronize(); // 同步设备，确保所有操作完成
    cudaError_t error = cudaGetLastError(); // 获取设备错误状态
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

   
    // 将数据传输到设备端
    cufftComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * NR * NP);
    cudaMemcpy(d_data, data.data(), sizeof(cufftComplex) * NR * NP, cudaMemcpyHostToDevice);

    // 检查cudaMemcpy是否成功
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // 创建FFT计划
    cufftHandle plan;
    cufftPlan1d(&plan, NP, CUFFT_C2C, 1); // 创建1D FFT计划

    // 分配设备内存
    cufftComplex* FFT_data;
    cudaMalloc((void**)&FFT_data, sizeof(cufftComplex) * NP * NP); //存储FFT后的数组
    cudaMemset(FFT_data, 0, sizeof(cufftComplex) * NP * NP);

    cufftComplex* p_data;
    cudaMalloc((void**)&p_data, sizeof(cufftComplex) * NP * NP); //存储FFTshift后的数组
    cudaMemset(p_data, 0, sizeof(cufftComplex) * NP * NP );

    cufftComplex* normalized_data = new cufftComplex[NR * NP];
    cudaMalloc((void**)&normalized_data, sizeof(cufftComplex) * NP * NP); //存储FFTshift后的数组
    cudaMemset(normalized_data, 0, sizeof(cufftComplex) * NP * NP);

    float* d_magnitude_data;
    cudaMalloc((void**)&d_magnitude_data, sizeof(float) * NP * NP); //存储FFT后的数组
    cudaMemset(d_magnitude_data, 0, sizeof(float) * NP * NP); // 将数组初始化为零



    cufftComplex* d_buffer;
    cudaMalloc((void**)&d_buffer, sizeof(cufftComplex) * NP); // 为一行数据分配内存

    float* d_magnitude;
    cudaMalloc((void**)&d_magnitude, sizeof(float) * NP); // 只为一行数据的幅度分配内存

    float* d_maxVal;
    cudaMalloc((void**)&d_maxVal, sizeof(float)); // 用于存储最大绝对值

    for (int i = 0; i < NR; ++i) {
        // 将一行数据从主机复制到设备的临时缓冲区
        cudaMemcpy(d_buffer, d_data + i * NP, sizeof(cufftComplex) * NP, cudaMemcpyHostToDevice);

        // 执行FFT
        cufftExecC2C(plan, d_buffer, d_buffer, CUFFT_FORWARD);
        cudaMemcpy(FFT_data + i * NP, d_buffer, sizeof(cufftComplex) * NP, cudaMemcpyDeviceToDevice);

        cudaDeviceSynchronize();

        // 执行FFTshift操作
       // fftShift << <1, NP >> > (d_buffer, NP, 1);
        fftshiftThrust(d_buffer,NP);

        // 将处理结果复制到设备内存的相应位置
        cudaMemcpy(p_data + i * NP, d_buffer, sizeof(cufftComplex) * NP, cudaMemcpyDeviceToDevice);

        //每行归一化操作
        float max_value = 178422.0f; // 设置固定的最大值  178422.0f改为17842.2f
        normalize(d_buffer, normalized_data + i * NP, max_value, NR);

        computeMagnitude(normalized_data + i * NP, d_magnitude_data + i * NP, NP);
    }
    cudaDeviceSynchronize();

    // 创建用于存储计算结果的临时数组
    cufftComplex* FFT_result = new cufftComplex[NR * NP];
    cufftComplex* h_result = new cufftComplex[NR * NP];
    cufftComplex* normalize_result = new cufftComplex[NR * NP];

    // 将结果从GPU拷贝到CPU内存
    //读取FFT后的数据
    cudaMemcpy(FFT_result, FFT_data, sizeof(cufftComplex) * NR * NP, cudaMemcpyDeviceToHost);
    const char* filename_FFT = "FFT.mat";
    saveMatToFile2(filename_FFT, FFT_result, NR, NP);

    //读取FFTshift后的数据
    cudaMemcpy(h_result, p_data, sizeof(cufftComplex) * NR * NP, cudaMemcpyDeviceToHost);
    const char* filename2 = "FFTshift.mat";
    saveMatToFile2(filename2, h_result,NR,NP);

    // 归一化数据
    cudaMemcpy(normalize_result, normalized_data, sizeof(cufftComplex) * NR * NP, cudaMemcpyDeviceToHost);
    const char* filename_guiyi = "guiyihua.mat";
    saveMatToFile2(filename_guiyi, normalize_result, NR, NP);

    // 在保存到文件之前，确保将设备上的数据传输回主机
    float* h_magnitude_data = new float[NP * NP];
    cudaMemcpy(h_magnitude_data, d_magnitude_data, sizeof(float) * NP * NP, cudaMemcpyDeviceToHost);

    // 然后使用您的 saveMatToFile 函数将数据保存到文件中
    const char* filename_jueduizhi = "jueduizhi.mat";
    saveMatToFile(filename_jueduizhi, h_magnitude_data, NP, NP);




   // // 计算绝对值???
   //int numBlocks = (NR * NP + 255) / 256;
   //complexMagnitude << <numBlocks, NP >> > (normalized_data, d_magnitude, NP, 1);
   //// CUDA kernel: 对数组中的每个元素应用函数20*log10(a[i])
   //cudaDeviceSynchronize();
   // 在主函数中调用这个内核
    //归一化在cv里面做
  /* int numBlocks1 = (NR * NP + 255) / 256;
   applyLog << <numBlocks1, 256 >> > (d_magnitude, NR * NP);
   cudaDeviceSynchronize();*/

    //有点错误先注释掉
    //// 计算绝对值后的数据的临时数组
    //float* normalized_magnitude_data = new float[NR * NP];

    //// 将处理后的数据复制回主机内存
    //for (int i = 0; i < NR; ++i) {
    //    cudaMemcpy(normalized_magnitude_data + i * NP, h_magnitude_data, sizeof(float) * NP, cudaMemcpyDeviceToHost);
    //}
    //const char* filename4 = "GT4.mat";
    //saveMatToFile(filename4, normalized_magnitude_data, NR, NP);
    //创建并显示图像

    cv::Mat img = cv::Mat::zeros(NR, NP, CV_32FC1);
    for (int i = 0; i < NR; ++i) {
        for (int j = 0; j < NP; ++j) {
            // img.at<float>(i, j) = h_data[i * NP + j].x;
            img.at<float>(i, j) = 20.0 * log10(h_magnitude_data[i * NP + j]);
        }
    }

    const char* filename5 = "Yak42_ISAR.mat";
    saveMatToFile3(filename5, img);

    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);

    // 保存图像文件
    cv::imwrite("Yak42_ISAR.png", img); // 将图像保存为 PNG 格式

    // Convert to 8-bit image
    cv::Mat img_8u;
    img.convertTo(img_8u, CV_8U);
    // 显示ISAR图像
    // 创建窗口
    cv::namedWindow("Yak42_ISAR", cv::WINDOW_AUTOSIZE);

    // 创建滑动条
    int slider_value = 10; // 初始值，对应1.0倍放大
    cv::createTrackbar("Zoom", "Yak42_ISAR", &slider_value, 100, on_trackbar, &img_8u);

    // 显示初始图像
    on_trackbar(slider_value, &img_8u);

    // 等待按键
    cv::waitKey(0);

    // 释放内存
    cudaFree(d_data);
    cudaFree(p_data);
    cudaFree(d_magnitude);
    cudaFree(d_maxVal);
    cufftDestroy(plan);
    // 释放主机内存
    delete[] h_result;
    // 释放临时数组的内存
    delete[] h_magnitude_data;
    



    return 0;
}

