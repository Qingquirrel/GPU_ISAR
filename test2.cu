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
    cv::log(ISARImg, ISARImg); // 对数变换

    // 将图像调整为与MATLAB相匹配的范围
    double minVal, maxVal;
    cv::minMaxLoc(ISARImg, &minVal, &maxVal);
    double targetMin = 20; // 设置目标最小值
    double targetMax = 20 + 40; // 设置目标最大值，对应 MATLAB 中的 [20, maxVal] 范围
    ISARImg = (ISARImg - minVal) / (maxVal - minVal) * (targetMax - targetMin) + targetMin;

    // 显示ISAR图像
    cv::imshow("ISAR Image", ISARImg);
    cv::waitKey(0);

    // 释放内存
    free(h_magnitude);
    cudaFree(d_data);
    cudaFree(d_magnitude);
    cufftDestroy(plan);
}

