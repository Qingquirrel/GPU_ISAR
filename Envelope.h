#ifndef ENVELOPEPROCESSOR_H
#define ENVELOPEPROCESSOR_H

#include <vector>
#include "mythrust.h"
#include "yak.h"
#define M_PI 3.14159265358979323846
#define I std::complex<float>(0.0f, 1.0f)

__global__ void computeEnvelope(const cufftComplex* input, float* output, int width, int height);
__global__ void computeEvery25thRowEnvelope(const cufftComplex* input, float* output, int width, int height);
__global__ void preprocessKernel(cufftComplex* data, int width, int height, float* hammingWindow, float* phaseShift);
__global__ void phaseCompensationKernel(cufftComplex* data, float* vecN, int width, float mopt);
__global__ void updateTemplateKernel(cufftComplex* templateData, cufftComplex* currentData, int width, float alpha);
__global__ void convertToComplex(float* input, cufftComplex* output, int size);
__global__ void computeAbs(cufftComplex* input, float* output, int size);
__global__ void fftShift(cufftComplex* data, int width, int height);
__global__ void ifftShiftKernel(cufftComplex* data, int width, int height);


class EnvelopeProcessor {
public:
	// Constructor and Destructor
	EnvelopeProcessor(int width, int height);
	~EnvelopeProcessor();

	void Envelope_show(const std::vector<cufftComplex>& input, cv::Mat& output);
	void computeAndShowEvery25thRowEnvelope(const std::vector<cufftComplex>& input);
	void processAndAlign(const std::vector<cufftComplex>& input, cv::Mat& output);
	void Envelope_show2(const std::vector<cufftComplex>& input);

	


private:
	int width_;
	int height_;
	cufftComplex* d_input_;// Device memory for input data
	float* d_output_;      // Device memory for output data
	cufftComplex* d_template_; // Device memory for template data
	float* d_hammingWindow_; // Device memory for Hamming window
	float* d_phaseShift_; // Device memory for phase shift
	float* d_vecN_; // Device memory for linear phase vector
	std::vector<float> h_hammingWindow_; // Host memory for Hamming window
	//std::vector<float> h_phaseShift_; // Host memory for phase shift
	std::vector<std::complex<float>> h_phaseShift_;
	std::vector<float> h_vecN_; // Host memory for linear phase vector
	std::vector<float> h_output_; // Host memory for output data


};

#endif // ENVELOPEPROCESSOR_H
