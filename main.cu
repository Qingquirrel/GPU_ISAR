//#include "yak.h"
//#include "mythrust.h"
//#include <vector>
#include <matio.h>
#include "Envelope.h"

typedef thrust::complex<float> comThr;


#define NR 256 // �״����ݵ�����
#define NP 256 // �״����ݵ�����





// CUDA kernel: ִ��fftshift����
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


// CUDA kernel: ��һ����������
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
//��������ͼ
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

    // ���� hamming ���� Ps
    for (int i = 0; i < N; ++i) {
        h_hamming[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (N - 1));
        h_Ps[i] = h_hamming[i] * thrust::exp(thrust::complex<float>(0, -M_PI * i));
        h_Vec_N[i] = i;
    }

    // ���䵽 GPU
    d_hamming = h_hamming;
    d_Ps = h_Ps;
    d_Vec_N = h_Vec_N;
}



//��ȡʵ��
void saveMatToFile(const char* filename, const float* data, int nr, int np) {
    MATFile* pmat;
    mxArray* pa;
    pmat = matOpen(filename, "w");
    if (pmat == NULL) {
        std::cerr << "Error creating file " << filename << std::endl;
        return;
    }

    mwSize dims[2] = { nr, np };
    pa = mxCreateNumericMatrix(nr, np, mxSINGLE_CLASS, mxREAL); // �޸���������Ϊ mxREAL
    if (pa == NULL) {
        std::cerr << "Error creating mxArray" << std::endl;
        matClose(pmat);
        return;
    }

    // ��ʵ�����ݿ����� mxArray ��
    float* pData = reinterpret_cast<float*>(mxGetData(pa));
    std::memcpy(pData, data, sizeof(float) * nr * np);

    matPutVariable(pmat, "data", pa);
    matClose(pmat);
}
//��ȡ����
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

    // ���������ݿ����� mxArray ��
    cufftComplex* pComplex = reinterpret_cast<cufftComplex*>(mxGetData(pa));
    for (int i = 0; i < nr * np; ++i) {
        pComplex[i].x = data[i].x;
        pComplex[i].y = data[i].y;
    }

    matPutVariable(pmat, "data", pa);
    matClose(pmat);
}
//��ȡcv����
void saveMatToFile3(const char* filename, const cv::Mat& mat) {
    // �� MAT �ļ�
    MATFile* pmat = matOpen(filename, "w");
    if (pmat == NULL) {
        std::cerr << "Error creating file " << filename << std::endl;
        return;
    }

    // ���� mxArray
    mxArray* pa = mxCreateNumericMatrix(mat.rows, mat.cols, mxSINGLE_CLASS, mxREAL);
    if (pa == NULL) {
        std::cerr << "Error creating mxArray" << std::endl;
        matClose(pmat);
        return;
    }

    // �����ݿ����� mxArray
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

    // �� mxArray д�� MAT �ļ�
    matPutVariable(pmat, "data", pa);
    matClose(pmat);
}
//���ж�ȡ
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

// �ص����������ڴ��������仯
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

// ת�ø�������
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

    // �����ڴ����ڴ洢����
    std::vector<cufftComplex> data(NR * NP);
    std::vector<cufftComplex> data_zhuanzhi(NR * NP);

    // ��ȡ���ݲ�����洢��������
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
      // ת�þ���
    // ת�ú������
    std::vector<cufftComplex> transposedInput(NR * NP);
    transposeComplexMatrix(data, transposedInput, NR, NP);

    // ����CUDA�¼�
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // ��¼��ʼ�¼�
    cudaEventRecord(start, 0);

    Processor.processAndAlign(transposedInput, envelope2_image);

    // ��¼�����¼�
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // ����ʱ���
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // ��ӡִ��ʱ��
    std::cout << "processAndAlign ִ��ʱ��: " << elapsedTime << " ����" << std::endl;
    // ����CUDA�¼�
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::vector<cufftComplex> juliduiqi_data(NR * NP);
    transposeComplexMatrix(transposedInput, juliduiqi_data, NR, NP);

    Processor.Envelope_show(juliduiqi_data, envelope3_image);

    // ���� MATLAB ��ͼ�������߼�����ͼ��
    for (int l = 0; l < NP; l += 50) {
        cv::Mat temp;
        cv::normalize(envelope2_image.row(l), temp, 0, 255, cv::NORM_MINMAX);
        cv::line(envelope2_image, cv::Point(0, l), cv::Point(NR - 1, l), cv::Scalar(255, 0, 0), 2);
        envelope2_image.row(l) = envelope2_image.row(l) + l;
    }

    cv::imshow("Aligned Data", envelope2_image);
    cv::waitKey(0);
    ////////////////////////////////////////////////




    // ��ӡǰ����������
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_data[" << i << "] = " << data[i].x << " + " << data[i].y << "i" << std::endl;
    }

    const char* filename1 = "Copy_of_Yak42_dat.mat";
    saveMatToFile2(filename1, data.data(), NR, NP);

    cudaDeviceReset(); // ����CUDA�豸״̬
    cudaSetDevice(0); // ����CUDA�豸Ϊ0���豸
    cudaDeviceSynchronize(); // ͬ���豸��ȷ�����в������
    cudaError_t error = cudaGetLastError(); // ��ȡ�豸����״̬
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

   
    // �����ݴ��䵽�豸��
    cufftComplex* d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * NR * NP);
    cudaMemcpy(d_data, data.data(), sizeof(cufftComplex) * NR * NP, cudaMemcpyHostToDevice);

    // ���cudaMemcpy�Ƿ�ɹ�
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }

    // ����FFT�ƻ�
    cufftHandle plan;
    cufftPlan1d(&plan, NP, CUFFT_C2C, 1); // ����1D FFT�ƻ�

    // �����豸�ڴ�
    cufftComplex* FFT_data;
    cudaMalloc((void**)&FFT_data, sizeof(cufftComplex) * NP * NP); //�洢FFT�������
    cudaMemset(FFT_data, 0, sizeof(cufftComplex) * NP * NP);

    cufftComplex* p_data;
    cudaMalloc((void**)&p_data, sizeof(cufftComplex) * NP * NP); //�洢FFTshift�������
    cudaMemset(p_data, 0, sizeof(cufftComplex) * NP * NP );

    cufftComplex* normalized_data = new cufftComplex[NR * NP];
    cudaMalloc((void**)&normalized_data, sizeof(cufftComplex) * NP * NP); //�洢FFTshift�������
    cudaMemset(normalized_data, 0, sizeof(cufftComplex) * NP * NP);

    float* d_magnitude_data;
    cudaMalloc((void**)&d_magnitude_data, sizeof(float) * NP * NP); //�洢FFT�������
    cudaMemset(d_magnitude_data, 0, sizeof(float) * NP * NP); // �������ʼ��Ϊ��



    cufftComplex* d_buffer;
    cudaMalloc((void**)&d_buffer, sizeof(cufftComplex) * NP); // Ϊһ�����ݷ����ڴ�

    float* d_magnitude;
    cudaMalloc((void**)&d_magnitude, sizeof(float) * NP); // ֻΪһ�����ݵķ��ȷ����ڴ�

    float* d_maxVal;
    cudaMalloc((void**)&d_maxVal, sizeof(float)); // ���ڴ洢������ֵ

    for (int i = 0; i < NR; ++i) {
        // ��һ�����ݴ��������Ƶ��豸����ʱ������
        cudaMemcpy(d_buffer, d_data + i * NP, sizeof(cufftComplex) * NP, cudaMemcpyHostToDevice);

        // ִ��FFT
        cufftExecC2C(plan, d_buffer, d_buffer, CUFFT_FORWARD);
        cudaMemcpy(FFT_data + i * NP, d_buffer, sizeof(cufftComplex) * NP, cudaMemcpyDeviceToDevice);

        cudaDeviceSynchronize();

        // ִ��FFTshift����
       // fftShift << <1, NP >> > (d_buffer, NP, 1);
        fftshiftThrust(d_buffer,NP);

        // �����������Ƶ��豸�ڴ����Ӧλ��
        cudaMemcpy(p_data + i * NP, d_buffer, sizeof(cufftComplex) * NP, cudaMemcpyDeviceToDevice);

        //ÿ�й�һ������
        float max_value = 178422.0f; // ���ù̶������ֵ  178422.0f��Ϊ17842.2f
        normalize(d_buffer, normalized_data + i * NP, max_value, NR);

        computeMagnitude(normalized_data + i * NP, d_magnitude_data + i * NP, NP);
    }
    cudaDeviceSynchronize();

    // �������ڴ洢����������ʱ����
    cufftComplex* FFT_result = new cufftComplex[NR * NP];
    cufftComplex* h_result = new cufftComplex[NR * NP];
    cufftComplex* normalize_result = new cufftComplex[NR * NP];

    // �������GPU������CPU�ڴ�
    //��ȡFFT�������
    cudaMemcpy(FFT_result, FFT_data, sizeof(cufftComplex) * NR * NP, cudaMemcpyDeviceToHost);
    const char* filename_FFT = "FFT.mat";
    saveMatToFile2(filename_FFT, FFT_result, NR, NP);

    //��ȡFFTshift�������
    cudaMemcpy(h_result, p_data, sizeof(cufftComplex) * NR * NP, cudaMemcpyDeviceToHost);
    const char* filename2 = "FFTshift.mat";
    saveMatToFile2(filename2, h_result,NR,NP);

    // ��һ������
    cudaMemcpy(normalize_result, normalized_data, sizeof(cufftComplex) * NR * NP, cudaMemcpyDeviceToHost);
    const char* filename_guiyi = "guiyihua.mat";
    saveMatToFile2(filename_guiyi, normalize_result, NR, NP);

    // �ڱ��浽�ļ�֮ǰ��ȷ�����豸�ϵ����ݴ��������
    float* h_magnitude_data = new float[NP * NP];
    cudaMemcpy(h_magnitude_data, d_magnitude_data, sizeof(float) * NP * NP, cudaMemcpyDeviceToHost);

    // Ȼ��ʹ������ saveMatToFile ���������ݱ��浽�ļ���
    const char* filename_jueduizhi = "jueduizhi.mat";
    saveMatToFile(filename_jueduizhi, h_magnitude_data, NP, NP);




   // // �������ֵ???
   //int numBlocks = (NR * NP + 255) / 256;
   //complexMagnitude << <numBlocks, NP >> > (normalized_data, d_magnitude, NP, 1);
   //// CUDA kernel: �������е�ÿ��Ԫ��Ӧ�ú���20*log10(a[i])
   //cudaDeviceSynchronize();
   // ���������е�������ں�
    //��һ����cv������
  /* int numBlocks1 = (NR * NP + 255) / 256;
   applyLog << <numBlocks1, 256 >> > (d_magnitude, NR * NP);
   cudaDeviceSynchronize();*/

    //�е������ע�͵�
    //// �������ֵ������ݵ���ʱ����
    //float* normalized_magnitude_data = new float[NR * NP];

    //// �����������ݸ��ƻ������ڴ�
    //for (int i = 0; i < NR; ++i) {
    //    cudaMemcpy(normalized_magnitude_data + i * NP, h_magnitude_data, sizeof(float) * NP, cudaMemcpyDeviceToHost);
    //}
    //const char* filename4 = "GT4.mat";
    //saveMatToFile(filename4, normalized_magnitude_data, NR, NP);
    //��������ʾͼ��

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

    // ����ͼ���ļ�
    cv::imwrite("Yak42_ISAR.png", img); // ��ͼ�񱣴�Ϊ PNG ��ʽ

    // Convert to 8-bit image
    cv::Mat img_8u;
    img.convertTo(img_8u, CV_8U);
    // ��ʾISARͼ��
    // ��������
    cv::namedWindow("Yak42_ISAR", cv::WINDOW_AUTOSIZE);

    // ����������
    int slider_value = 10; // ��ʼֵ����Ӧ1.0���Ŵ�
    cv::createTrackbar("Zoom", "Yak42_ISAR", &slider_value, 100, on_trackbar, &img_8u);

    // ��ʾ��ʼͼ��
    on_trackbar(slider_value, &img_8u);

    // �ȴ�����
    cv::waitKey(0);

    // �ͷ��ڴ�
    cudaFree(d_data);
    cudaFree(p_data);
    cudaFree(d_magnitude);
    cudaFree(d_maxVal);
    cufftDestroy(plan);
    // �ͷ������ڴ�
    delete[] h_result;
    // �ͷ���ʱ������ڴ�
    delete[] h_magnitude_data;
    



    return 0;
}

