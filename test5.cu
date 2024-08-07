#include "yak.h"
#include "mythrust.h"



#define NR 256 // 雷达数据的行数
#define NP 256 // 雷达数据的列数

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



// CUDA kernel: 将归一化的幅度图转换成热力图，并将其放置在指定的坐标范围内
__global__ void generateHeatmap(float* d_input, uchar3* d_output, float minVal, float maxVal, float x_min, float x_max, float y_min, float y_max, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        // 归一化到 [0, 1]
        float normalizedValue = (d_input[y * width + x] - minVal) / (maxVal - minVal);
        // 将归一化后的值映射到指定的坐标范围
        float x_coord = x_min + (x_max - x_min) * x / static_cast<float>(width);
        float y_coord = y_min + (y_max - y_min) * y / static_cast<float>(height);
        // 计算热力图颜色
        uchar3 color = make_uchar3(static_cast<unsigned char>(255 * normalizedValue), 0, 0); // 红色通道表示幅度
        d_output[y * width + x] = color;
    }
}

void saveMatToFile(const char* filename, const cufftComplex* data, int nr, int np) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    // 写入数据到文件
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < np; ++j) {
            file << data[i * np + j].x << "+" << data[i * np + j].y << "i ";
        }
        file << "\n";
    }

    file.close();
}
#include <opencv2/opencv.hpp>

#include <mat.h>

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







// 生成ISAR图像
void generateISAR(const cufftComplex* h_data, int nr, int np, float prf, float dr) {


    //二维FFT计划
    cufftComplex* CompData = (cufftComplex*)malloc(256*256 * sizeof(cufftComplex));
    for (unsigned i = 0; i < 256 * 256; i++) {
        CompData[i].x = h_data[i].x;
        CompData[i].y = h_data[i].y;
    }

    cufftComplex* d_fftData;
    cudaMalloc((void**)&d_fftData, 256 * 256 * sizeof(cufftComplex));
    cudaMemcpy(d_fftData, CompData, 256 * 256 * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan2d(&plan, 256, 256, CUFFT_C2C);
    cufftExecC2C(plan, (cufftComplex*)d_fftData, (cufftComplex*)d_fftData, CUFFT_FORWARD); cudaDeviceSynchronize();


    // 从设备内存中读取 d_fftData 数据
    cudaMemcpy(CompData, d_fftData, 256 * 256 * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    const char* filename = "fft_data4.mat";
    saveMatToFile2(filename, CompData, nr, np);

   
    dim3 blockSize(nr, 1); // 修改blockSize，使得每个block包含一行
    dim3 gridSize(1, nr); // 修改gridSize，使得grid包含nr个block
    fftShift<< <gridSize, blockSize >> > (d_fftData, np, nr);


    // 计算绝对值
    float* d_magnitude;
    cudaMalloc((void**)&d_magnitude, sizeof(float) * nr * np);
    complexMagnitude << <gridSize, blockSize >> > (d_fftData, d_magnitude, np, nr);

    // 复制数据从设备到主机
    float* h_magnitude1;
    h_magnitude1 = (float*)malloc(sizeof(float) * nr * np);
    cudaMemcpy(h_magnitude1, d_magnitude, sizeof(float) * nr * np, cudaMemcpyDeviceToHost);



    // 创建ISAR图像
    cv::Mat ISARImg(nr, np, CV_32FC1, h_magnitude1);

    // 计算幅度
    cv::Mat magnitude;
    cv::absdiff(ISARImg, cv::Scalar(0), magnitude);


    // 对数变换
    cv::log(magnitude, magnitude);
    ISARImg = 20 * magnitude;

    // 归一化
    cv::Mat normalizedISARImg;
    cv::normalize(ISARImg, normalizedISARImg, 0, 255, cv::NORM_MINMAX, CV_8UC1); // 将单通道的ISAR图像归一化为8位灰度图像
    cv::imwrite("normalized_ISAR_img.png", normalizedISARImg);

    // 释放幅度图内存
    magnitude.release();

    // 创建热力图
    uchar3* d_heatmap;
    cudaMalloc((void**)&d_heatmap, sizeof(uchar3) * nr * np);

    // 指定坐标范围
    float x_min = -prf / 2.0f;
    float x_max = prf / 2.0f;
    float y_min = -dr * nr / 2.0f;
    float y_max = dr * nr / 2.0f;

    // 调用CUDA kernel生成热力图
    generateHeatmap << <gridSize, blockSize >> > (reinterpret_cast<float*>(normalizedISARImg.data), d_heatmap, 0, 255, x_min, x_max, y_min, y_max, np, nr);

    // 将热力图拷贝回主机内存
    uchar3* h_heatmap = new uchar3[nr * np];
    cudaMemcpy(h_heatmap, d_heatmap, sizeof(uchar3) * nr * np, cudaMemcpyDeviceToHost);

    // 将热力图数据保存为MATLAB格式（.mat）
    cv::Mat heatmapMat(np, nr, CV_8UC3, h_heatmap);
    cv::Mat heatmapMatFloat;
    heatmapMat.convertTo(heatmapMatFloat, CV_32FC3); // 转换为浮点型

    cv::FileStorage fs("heatmap_data.yml", cv::FileStorage::WRITE);
    fs << "heatmap" << heatmapMatFloat;
    fs.release();

    // 将热力图数据保存为图片
    cv::imwrite("heatmap_data.png", heatmapMat);

    // 创建窗口并绘制热力图
    cv::namedWindow("Heatmap with Axes", cv::WINDOW_GUI_NORMAL);
    cv::imshow("Heatmap with Axes", heatmapMat);

    // 等待按键
    cv::waitKey(0);

    // 释放内存
    delete[] h_magnitude1;
    delete[] h_heatmap;
    cudaFree(d_fftData);
    cudaFree(d_magnitude);
    cudaFree(d_heatmap);
    cufftDestroy(plan);
}

int main() {
   float c = 2.99792458e8;
   float drs = 1.0;
   float bw = 6e9 / drs;
   float Pri = 120e-6;
   float prf = 1 / Pri;

   // 读取MATLAB文件中的数据
   MATFile* pmat;
   mxArray* pmxArray;
   pmat = matOpen("D:\\AAA_hot_pot\\simple_isar_image\\Yak42\\Data\\Yak42_transposed.mat", "r");
   if (pmat == NULL) {
       std::cerr << "Error opening file." << std::endl;
       return -1;
   }

   pmxArray = matGetVariable(pmat, "y_transposed");
   if (pmxArray == NULL) {
       std::cerr << "Error reading variable 'y_transposed' from file." << std::endl;
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


// 打开.mat文件
    mat_t* matfp = Mat_Open("D:\\AAA_hot_pot\\GPU_image\\yak42_ISAR\\Yak42_transposed.mat", MAT_ACC_RDONLY);
    if (matfp == NULL) {
        std::cerr << "Error opening MAT file" << std::endl;
        return -1;
    }
    else {
        std::cout << "MAT file opened successfully" << std::endl;
    }


    // 读取变量
    matvar_t* matvar = Mat_VarRead(matfp, "y_transposed"); // 请将"your_variable_name"替换为您的变量名
    if (matvar == NULL) {
        std::cerr << "Error reading variable from MAT file" << std::endl;
        return -1;
    }
    else {
        std::cout << "Variable read successfully" << std::endl;
    }

    std::cout << "Data array contents:" << std::endl;
    for (int i = 0; i < matvar->nbytes / sizeof(double); ++i) {
        std::cout << "data[" << i << "] = " << ((double*)matvar->data)[i] << std::endl;
    }

    // 确保变量是复数矩阵
    if (matvar->class_type != MAT_C_DOUBLE || matvar->isComplex == 0) {
        std::cerr << "Variable is not a complex matrix" << std::endl;
        return -1;
    }

    // 分配CUDA内存
    cufftComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cuDoubleComplex) * matvar->dims[0] * matvar->dims[1]);

    // 将数据复制到CUDA内存
    memcpy(d_data, matvar->data, sizeof(cuDoubleComplex) * matvar->dims[0] * matvar->dims[1]);

    // 检查cudaMemcpy是否成功
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // 打印前几个数据项
    std::cout << "First few data items:" << std::endl;
    int num_items_to_print = 10; // 要打印的数据项数目
    for (int i = 0; i < num_items_to_print; ++i) {
        std::cout << "Data[" << i << "] = " << d_data[i].x << " + " << d_data[i].y << "i" << std::endl;
    }


    // 清理
    Mat_VarFree(matvar);
    Mat_Close(matfp);



std::ifstream file("D:\\AAA_hot_pot\\GPU_image\\yak42_ISAR\\Copy_of_Yak42.dat", std::ios::binary);
    if (!file) {
        std::cerr << "Error opening data file" << std::endl;
        return -1;
    }

    // 分配内存用于存储数据
    std::vector<cufftComplex> data(NR * NP);

    // 读取数据并将其存储在数组中
    for (int i = 0; i < NR * NP; ++i) {
        // 读取实部
        file.read(reinterpret_cast<char*>(&data[i].x), sizeof(float));
        if (!file) {
            std::cerr << "Error reading real part from file" << std::endl;
            return -1;
        }
        // 读取虚部
        file.read(reinterpret_cast<char*>(&data[i].y), sizeof(float));
        if (!file) {
            std::cerr << "Error reading imaginary part from file" << std::endl;
            return -1;
        }
    }

    const char* filename5 = "data1.mat";
    saveMatToFile(filename5, data.data(), NR, NP);


    // 打印data数据
    for (int i = 0; i <100; ++i) {
        std::cout << "data[" << i << "] = " << data[i].x << " + " << data[i].y << "i" << std::endl;
    }


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

void saveMatToFile(const char* filename, const cufftComplex* data, int nr, int np) {
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

// 创建并显示图像
//    cv::Mat img = cv::Mat::zeros(NR, NP, CV_32FC1);
//    for (int i = 0; i < NR; ++i) {
//        for (int j = 0; j < NP; ++j) {
//           // img.at<float>(i, j) = h_data[i * NP + j].x;
//            img.at<float>(i, j) = 20 * std::log10(h_result[i * NP + j].x);
//        }
//    }
//
//    const char* filename2 = "img11.mat";
//    saveMatToFile3(filename2, img);
//
//    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
//
//    // 保存图像文件
//    cv::imwrite("img11.png", img); // 将图像保存为 PNG 格式
//
//    // 读取图像文件
//    cv::Mat img_loaded = cv::imread("img11.png", cv::IMREAD_GRAYSCALE); // 读取图像文件（灰度图像）
//
//    // 将物理单位转换为像素单位
//    int prf_pixels = static_cast<int>(prf * img.cols / (2 * bw)); // prf在x方向上的像素数
//    int nr_pixels = static_cast<int>(NR * img.rows / (2 * dr)); // nr在y方向上的像素数
//
//    // 定义感兴趣区域（ROI）
//    cv::Rect roi(cv::Point(img.cols / 2 - prf_pixels / 2, img.rows / 2 - nr_pixels / 2), cv::Size(prf_pixels, nr_pixels));
//
//    // 检查 ROI 是否超出图像范围
//    cv::Rect imgRect(0, 0, img.cols, img.rows); // 创建图像范围的矩形
//    cv::Rect roiRect = roi & imgRect; // 计算 ROI 和图像范围的交集
//
//    // 检查 ROI 是否为空
//    if (roiRect.area() > 0) {
//        // 如果交集不为空，则说明 ROI 在图像范围内
//        cv::Mat img_roi = img(roiRect); // 获取感兴趣区域
//        // 在这里可以对 img_roi 进行进一步处理
//        cv::imshow("Cropped Image", img_roi); // 显示裁剪后的图像
//        cv::waitKey(0); // 等待按键
//    }
//    else {
//        // 如果交集为空，则说明 ROI 超出了图像范围
//        std::cerr << "ROI is out of image range." << std::endl;
//    }
//
//
//    // 将 ROI 应用到图像上
//    cv::Mat img_roi = img(roi);
//
//    // 显示裁剪后的图像
//    cv::imshow("Cropped Image", img_roi);
//    cv::waitKey(0); // 等待按键



    ///////求矩阵元素最大值测试模块/////////////
    // // 找到绝对值最大的元素
    //float max_abs_value = 0.0f; // 用于存储绝对值最大的元素的绝对值
    //int max_abs_index_i = 0; // 用于存储绝对值最大的元素的行索引
    //int max_abs_index_j = 0; // 用于存储绝对值最大的元素的列索引

    //for (int i = 0; i < NR; ++i) {
    //    for (int j = 0; j < NP; ++j) {
    //        // 计算当前元素的绝对值
    //        float abs_value = sqrtf(h_result[i * NP + j].x * h_result[i * NP + j].x +
    //            h_result[i * NP + j].y * h_result[i * NP + j].y);
    //        // 如果当前元素的绝对值大于目前记录的最大值，则更新最大值和索引
    //        if (abs_value > max_abs_value) {
    //            max_abs_value = abs_value;
    //            max_abs_index_i = i;
    //            max_abs_index_j = j;
    //        }
    //    }
    //}

    //// 打印找到的最大元素
    //std::cout << "绝对值最大的元素：" << std::endl;
    //std::cout << "索引 i: " << max_abs_index_i << ", 索引 j: " << max_abs_index_j << std::endl;
    //std::cout << "实部：" << h_result[max_abs_index_i * NP + max_abs_index_j].x << std::endl;
    //std::cout << "虚部：" << h_result[max_abs_index_i * NP + max_abs_index_j].y << std::endl;
    //std::cout << "绝对值：" << max_abs_value << std::endl;