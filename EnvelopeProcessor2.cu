#include "Envelope.h"
//包络对齐出图
__global__ void computeEnvelope(const cufftComplex* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float real = input[idx].x;
        float imag = input[idx].y;
        output[idx] = sqrtf(real * real + imag * imag);
    }
}
__global__ void computeEnvelope2(const cufftComplex* input, float* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        float real = input[x].x;
        float imag = input[x].y;
        output[x] = sqrtf(real * real + imag * imag);
    }
}
__global__ void computeEnvelope3(const cufftComplex* input, float* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        float real = input[x].x;
        float imag = input[x].y;
        output[x] = sqrtf(real * real + imag * imag);
    }
}
__global__ void fftShift(cufftComplex* data, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int halfWidth = width / 2;

    if (x < halfWidth) {
        int index1 = x;
        int index2 = x + halfWidth;
        cufftComplex temp = data[index1];
        data[index1] = data[index2];
        data[index2] = temp;
    }
}
__global__ void computeEvery25thRowEnvelope(const cufftComplex* input, float* output, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = row * width + col;

        if (row % 25 == 0) {
            float real = input[idx].x;
            float imag = input[idx].y;
            output[idx] = 0.001f * sqrtf(real * real + imag * imag) + row;
        }
        else {
            output[idx] = 0.0f;
        }
    }
}
__global__ void preprocessKernel(cufftComplex* data, int width, int height, float* hammingWindow, float* phaseShift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx % width;
    if (idx < width * height) {
        data[idx].x *= hammingWindow[col] * phaseShift[col];
        data[idx].y *= hammingWindow[col] * phaseShift[col];
    }
}

__global__ void phaseCompensationKernel(cufftComplex* data, float* vecN, int width, float mopt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        float angle = -2.0f * M_PI * vecN[idx] / width * mopt;
        float cosAngle = cosf(angle);
        float sinAngle = sinf(angle);
        float real = data[idx].x;
        float imag = data[idx].y;
        data[idx].x = real * cosAngle - imag * sinAngle;
        data[idx].y = real * cosAngle + imag * sinAngle;
    }
}

__global__ void updateTemplateKernel(cufftComplex* templateData, cufftComplex* currentData, int width, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        templateData[idx].x = alpha * templateData[idx].x + (1.0f - alpha) * abs(currentData[idx].x);
        templateData[idx].y = alpha * templateData[idx].y + (1.0f - alpha) * abs(currentData[idx].y);
    }
}

// FFT shift kernel
__global__ void ifftShiftKernel(cufftComplex* data, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        int halfWidth = width / 2;
        int halfHeight = height / 2;
        int newIdx = (idx + halfWidth) % width;
        int newIdy = (idy + halfHeight) % height;
        int oldIndex = idy * width + idx;
        int newIndex = newIdy * width + newIdx;

        cufftComplex temp = data[oldIndex];
        data[oldIndex] = data[newIndex];
        data[newIndex] = temp;
    }
}

// 计算复数幅值的内核
__global__ void computeAbs(cufftComplex* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        cufftComplex val = input[idx];
        output[idx] = sqrt(val.x * val.x + val.y * val.y);
    }
}
// 将幅值转换回复数形式的内核
__global__ void convertToComplex(float* input, cufftComplex* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx].x = input[idx];
        output[idx].y = 0.0f;
    }
}
// 自定义 cufftComplex 乘法运算符
struct cufftComplexMultiply {
    __host__ __device__
        cufftComplex operator()(const cufftComplex& a, const cufftComplex& b) const {
        cufftComplex result;
        result.x = a.x * b.x - a.y * b.y;
        result.y = a.x * b.y + a.y * b.x;
        return result;
    }
};
//void performIFFT(cufftComplex* data, int width, int height) {
//    cufftHandle plan;
//    cufftPlan1d(&plan, width, CUFFT_C2C, height);
//    for (int row = 0; row < height; ++row) {
//        cufftExecC2C(plan, &data[row * width], &data[row * width], CUFFT_INVERSE);
//    }
//    cufftDestroy(plan);
//}
// Perform IFFT for each row
void performIFFT(cufftComplex* data, int width, int height) {
    for (int row = 0; row < height; ++row) {
        cufftHandle plan;
        cufftPlan1d(&plan, width, CUFFT_C2C, 1);
        cufftExecC2C(plan, &data[row * width], &data[row * width], CUFFT_INVERSE);
        cufftDestroy(plan);
    }
}
// 调用函数
void preprocessData(cufftComplex* d_data, int width, int height) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 执行 ifftshift
    fftShift << <numBlocks, threadsPerBlock >> > (d_data, width, height);
    cudaDeviceSynchronize();

    // 执行 ifft
    performIFFT(d_data, width, height);
}
// 参数初始化

// Constructor: Initialize member variables and allocate device memory
EnvelopeProcessor::EnvelopeProcessor(int width, int height)
    : width_(width), height_(height), d_input_(nullptr), d_output_(nullptr), d_template_(nullptr),
    d_hammingWindow_(nullptr), d_phaseShift_(nullptr), d_vecN_(nullptr) {
    // Allocate device memory
    cudaMalloc(&d_input_, width_ * height_ * sizeof(cufftComplex));
    cudaMalloc(&d_template_, width_ * height_ * sizeof(cufftComplex));
    cudaMalloc(&d_output_, width_ * height_ * sizeof(float));
    cudaMalloc(&d_hammingWindow_, width_ * sizeof(float));
    cudaMalloc(&d_phaseShift_, width_ * sizeof(float));
    cudaMalloc(&d_vecN_, width_ * sizeof(float));
    h_output_.resize(width_ * height_);
    h_hammingWindow_.resize(width_);
    h_phaseShift_.resize(width_);
    h_vecN_.resize(width_);
    for (int i = 0; i < width_; ++i) {
        h_hammingWindow_[i] = 0.54f - 0.46f * cosf(2 * M_PI * i / (width_ - 1));
        //h_phaseShift_[i] = std::exp(std::complex<float>(0.0f, M_PI * i / width));
        h_phaseShift_[i] = exp(-std::complex<float>(0, M_PI * i));
        h_vecN_[i] = i;
    }
    cudaMemcpy(d_hammingWindow_, h_hammingWindow_.data(), width_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phaseShift_, h_phaseShift_.data(), width_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecN_, h_vecN_.data(), width_ * sizeof(float), cudaMemcpyHostToDevice);
}

// Destructor: Free device memory
EnvelopeProcessor::~EnvelopeProcessor() {
    // Free device memory
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaFree(d_hammingWindow_);
    cudaFree(d_phaseShift_);
    cudaFree(d_vecN_);
    cudaFree(d_template_);
}

void EnvelopeProcessor::Envelope_show(const std::vector<cufftComplex>& input, cv::Mat& output) {
    // Copy data from host to device
    cudaMemcpy(d_input_, input.data(), width_ * height_ * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Define CUDA block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width_ + blockDim.x - 1) / blockDim.x, (height_ + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    computeEnvelope << <gridDim, blockDim >> > (d_input_, d_output_, width_, height_);

    // Copy results from device to host
    cudaMemcpy(h_output_.data(), d_output_, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert the result to OpenCV Mat
    output = cv::Mat(height_, width_, CV_32F, h_output_.data());

    // Normalize the image to 0-255
    cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
    output.convertTo(output, CV_8U);

    // Rotate the image 90 degrees counterclockwise
    cv::rotate(output, output, cv::ROTATE_90_COUNTERCLOCKWISE);

    // Create a custom colormap for black background and red foreground
    cv::Mat colormap(256, 1, CV_8UC3);
    for (int i = 0; i < 256; ++i) {
        colormap.at<cv::Vec3b>(i) = cv::Vec3b(0, 0, i); // Red channel gradient
    }

    // Apply a color map to the image to create a heatmap
    cv::applyColorMap(output, output, colormap);

    // Display the envelope image
    cv::imshow("Envelope", output);

    // Wait for a key press
    cv::waitKey(0);

}
void EnvelopeProcessor::Envelope_show2(const std::vector<cufftComplex>& input) {
    // Copy data from host to device
    cudaMemcpy(d_input_, input.data(), width_ * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Perform FFT on the input data
    cufftHandle plan;
    cufftPlan1d(&plan, width_, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_input_, d_input_, CUFFT_FORWARD);

    // Perform FFTShift on the FFT result
    dim3 blockDim(16);
    dim3 gridDim((width_ + blockDim.x - 1) / blockDim.x);
    fftShift << <gridDim, blockDim >> > (d_input_, width_);
    cudaDeviceSynchronize();

    // Compute the absolute value of the FFT result
    computeEnvelope2 << <gridDim, blockDim >> > (d_input_, d_output_, width_);
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(h_output_.data(), d_output_, width_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Create an image to display the peak plot
    int imgHeight = 400;
    int imgWidth = 600; // Set a fixed width for the plot
    cv::Mat plotImg = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC3);
    plotImg.setTo(cv::Scalar(255, 255, 255)); // Set background to white

    // Normalize the output values to the image height
    float maxVal = *std::max_element(h_output_.begin(), h_output_.end());
    int maxSamples = 300; // Maximum sample points for scaling

    int horizontalOffset = (imgWidth - maxSamples) / 2; // Offset to center the plot

    for (int i = 0; i < width_; ++i) {
        int y = static_cast<int>((h_output_[i] / maxVal) * imgHeight);
        int x = horizontalOffset + static_cast<int>((i / static_cast<float>(width_)) * maxSamples);
        cv::line(plotImg, cv::Point(x, imgHeight - 1), cv::Point(x, imgHeight - 1 - y), cv::Scalar(255, 0, 0)); // Blue peak
    }

    // Add amplitude information on the vertical axis
    int numYLabels = 5;
    for (int i = 0; i < numYLabels; ++i) {
        int y = imgHeight - 1 - static_cast<int>((i * imgHeight / (numYLabels - 1)));
        std::string label = std::to_string(static_cast<int>(i * 4 * 10000 / (numYLabels - 1)));
        cv::putText(plotImg, label, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::line(plotImg, cv::Point(horizontalOffset, y), cv::Point(imgWidth - 1, y), cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }

    // Add sampling points information on the horizontal axis
    int numXLabels = 6;
    for (int i = 0; i < numXLabels; ++i) {
        int x = horizontalOffset + static_cast<int>((i * maxSamples / (numXLabels - 1)));
        std::string label = std::to_string(static_cast<int>(i * maxSamples / (numXLabels - 1)));
        cv::putText(plotImg, label, cv::Point(x, imgHeight - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // Add vertical axis label (Amplitude)
    cv::putText(plotImg, "Amplitude", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    // Add horizontal axis label (Sample Points) at the bottom right
    cv::putText(plotImg, "Sample Points", cv::Point(imgWidth - 130, imgHeight - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    // Display the peak plot
    cv::imshow("Absolute Value Peak Plot of First Row", plotImg);

    // Wait for a key press
    cv::waitKey(0);

    // Cleanup
    cufftDestroy(plan);
}
void EnvelopeProcessor::computeAndShowEvery25thRowEnvelope(const std::vector<cufftComplex>& input) {
    // Copy data from host to device
    cudaMemcpy(d_input_, input.data(), width_ * height_ * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    // Define CUDA block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width_ + blockDim.x - 1) / blockDim.x, (height_ + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel to compute envelope for every 25th row
    computeEvery25thRowEnvelope << <gridDim, blockDim >> > (d_input_, d_output_, width_, height_);

    // Copy results from device to host
    cudaMemcpy(h_output_.data(), d_output_, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);

    // Create Mat object and draw envelope image
    cv::Mat envelope(height_, width_, CV_32F, h_output_.data());
    cv::normalize(envelope, envelope, 0, 255, cv::NORM_MINMAX);
    envelope.convertTo(envelope, CV_8U);

    // Rotate the image 90 degrees
    cv::rotate(envelope, envelope, cv::ROTATE_90_COUNTERCLOCKWISE);

    // Create a custom colormap for black background and red foreground
    cv::Mat colormap(256, 1, CV_8UC3);
    for (int i = 0; i < 256; ++i) {
        colormap.at<cv::Vec3b>(i) = cv::Vec3b(0, 0, i); // Red channel gradient
    }

    // Apply a color map to the image to create a heatmap
    cv::applyColorMap(envelope, envelope, colormap);

    // Display the envelope image
    cv::imshow("Every 25th Row Envelope", envelope);
    cv::waitKey(0);

}
void EnvelopeProcessor::processAndAlign(const std::vector<cufftComplex>& input, cv::Mat& output) {
    cudaMemcpy(d_input_, input.data(), width_ * height_ * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    // FFT shift
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width_ + threadsPerBlock.x - 1) / threadsPerBlock.x, (height_ + threadsPerBlock.y - 1) / threadsPerBlock.y);
    ifftShiftKernel << <numBlocks, threadsPerBlock >> > (d_input_, width_, height_);
    cudaDeviceSynchronize();

    // Perform IFFT for each row
    performIFFT(d_input_, width_, height_);

    //// 创建 FFT 计划
    //cufftHandle plan;
    //cufftPlan1d(&plan, width_, CUFFT_C2C, height_);

    // Process each row
   // Process each row

      // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始事件
    cudaEventRecord(start, 0);

    for (int row = 0; row < height_; ++row) {
        cufftComplex* rowData = &d_input_[row * width_];

        if (row == 0) {
            cufftHandle plan;
            cufftPlan1d(&plan, width_, CUFFT_C2C, 1);
            cufftExecC2C(plan, rowData, rowData, CUFFT_INVERSE);
            cudaMemcpy(d_template_, rowData, width_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
            cufftDestroy(plan);
        }
        else {
            cufftHandle plan;
            cufftPlan1d(&plan, width_, CUFFT_C2C, 1);
            cufftExecC2C(plan, d_template_, d_template_, CUFFT_FORWARD);

            thrust::device_vector<float> d_abs(width_);
            computeAbs << <numBlocks, threadsPerBlock >> > (rowData, thrust::raw_pointer_cast(d_abs.data()), width_);
            cudaDeviceSynchronize();

            thrust::device_vector<cufftComplex> d_abs_complex(width_);
            convertToComplex << <numBlocks, threadsPerBlock >> > (thrust::raw_pointer_cast(d_abs.data()), thrust::raw_pointer_cast(d_abs_complex.data()), width_);
            cudaDeviceSynchronize();

            cufftExecC2C(plan, thrust::raw_pointer_cast(d_abs_complex.data()), rowData, CUFFT_FORWARD);

            thrust::device_vector<cufftComplex> d_conj(width_);
            thrust::transform(thrust::device, rowData, rowData + width_, d_template_, d_conj.begin(), cufftComplexMultiply());
            cufftExecC2C(plan, thrust::raw_pointer_cast(d_conj.data()), thrust::raw_pointer_cast(d_conj.data()), CUFFT_INVERSE);

            thrust::host_vector<float> h_R(width_);
            cudaMemcpy(h_R.data(), thrust::raw_pointer_cast(d_conj.data()), width_ * sizeof(float), cudaMemcpyDeviceToHost);

            float maxR = *thrust::max_element(h_R.begin(), h_R.end());
            int maxm = thrust::distance(h_R.begin(), thrust::max_element(h_R.begin(), h_R.end()));

            float xstar = 0;
            if ((maxm != 0) && (maxm != width_ - 1)) {
                float f1 = h_R[maxm - 1];
                float f2 = h_R[maxm];
                float f3 = h_R[maxm + 1];
                float fa = (f1 + f3 - 2 * f2) / 2;
                float fb = (f3 - f1) / 2;
                xstar = -fb / (2 * fa);
            }

            float mopt = (maxm + xstar - 1) - (width_ / 2);

            phaseCompensationKernel << <numBlocks, threadsPerBlock >> > (rowData, d_vecN_, width_, mopt);
            cufftExecC2C(plan, rowData, rowData, CUFFT_INVERSE);
            cudaMemcpy(d_template_, rowData, width_ * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
            cufftDestroy(plan);
        }
    }

    // 记录结束事件
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算时间差
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // 打印执行时间
    std::cout << "CUDA 代码执行时间: " << elapsedTime << " 毫秒" << std::endl;

    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 提取处理后的数据
    cudaMemcpy(h_output_.data(), d_input_, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);

    output.create(height_, width_, CV_32FC1);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            output.at<float>(y, x) = abs(h_output_[y * width_ + x]);
        }
    }
}






