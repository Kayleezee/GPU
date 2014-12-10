/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : gpucomp02
 *
 *                   File : main.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>

const static int DEFAULT_MATRIX_SIZE = 1024;
const static int DEFAULT_BLOCK_DIM   =  128;

//
// Function Prototypes
//
void printHelp(char *);

//
// CPU
//
float cpu_reduce(float *pArray, int iSize) {
	int i;

	float fRes = 0.0;

	for(i = 0; i < iSize; i++) {
		fRes += pArray[i];
	}
	return fRes;
}


//
// Reduction_Kernel_Naive
//
__global__ void
reduction_KernelNaive(int numElements, float* dataIn, float* dataOut)
{
    extern __shared__ float sPartArray[];

    const int tid = threadIdx.x;
	unsigned int elementId = blockIdx.x * blockDim.x + threadIdx.x;

    sPartArray[tid] = dataIn[elementId];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        if(tid % (2 * s) == 0) {
            sPartArray[tid] += sPartArray[tid + s];
        }
        __syncthreads();
    }

	if (tid == 0) {
        dataOut[blockIdx.x] = sPartArray[0];
	}
}

//
// Reduction_Kernel_Optimized
//
__global__ void
reduction_KernelOptimized(int numElements, float* dataIn, float* dataOut)
{
    extern __shared__ float sPartArray[];

    const int tid = threadIdx.x;
	unsigned int elementId = blockIdx.x * blockDim.x + threadIdx.x;

    sPartArray[tid] = dataIn[elementId];
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) {
            sPartArray[tid] += sPartArray[tid + s];
        }
        __syncthreads();
    }

    /*
    extern __shared__ float sPartArray[];

    const int tid = threadIdx.x;


	unsigned int elementId = blockIdx.x * (blockDim.x * 2) + threadIdx.x;







    sPartArray[tid] = dataIn[elementId] + dataIn[elementId + blockDim.x];
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if(tid < s) {
            sPartArray[tid] += sPartArray[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32 && blockDim.x >= 64) sPartArray[tid] += sPartArray[tid + 32];
    if(tid < 16 && blockDim.x >= 32) sPartArray[tid] += sPartArray[tid + 16];
    if(tid <  8 && blockDim.x >= 16) sPartArray[tid] += sPartArray[tid +  8];
    if(tid <  4 && blockDim.x >=  8) sPartArray[tid] += sPartArray[tid +  4];
    if(tid <  2 && blockDim.x >=  4) sPartArray[tid] += sPartArray[tid +  2];
    if(tid <  1 && blockDim.x >=  2) sPartArray[tid] += sPartArray[tid +  1];
    */
	if (tid == 0) {
        dataOut[blockIdx.x] = sPartArray[0];
	}
}


//
// Main
//
int
main(int argc, char * argv[])
{
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp)
	{
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp)
	{
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer cpuTimer, kernelTimer, kernelOptTimer;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ?
			numElements : DEFAULT_MATRIX_SIZE;
	//
	// Host Memory
	//
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory)
	{
		pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
	}

	float* h_dataIn = NULL;
	float* h_dataOut = NULL;
	if (!pinnedMemory)
	{
		// Pageable
		h_dataIn = static_cast<float*>
				(malloc(static_cast<size_t>(numElements * sizeof(*h_dataIn))));
		h_dataOut = static_cast<float*>
				(malloc(static_cast<size_t>(sizeof(*h_dataOut))));
	}
	else
	{
		// Pinned
		cudaMallocHost(&h_dataIn,
				static_cast<size_t>(numElements * sizeof(*h_dataIn)));
		cudaMallocHost(&h_dataOut,
				static_cast<size_t>(sizeof(*h_dataOut)));
	}
	// Init h_dataIn & h_dataOut
	for(int i = 0; i < numElements; i++) {
		h_dataIn[i] = (float)1;
	}

	*h_dataOut = 0;


	/* CPU REDUCTION **************************************************/
	float fCPURes;

	cpuTimer.start();
	fCPURes = cpu_reduce(h_dataIn, numElements);
	cpuTimer.stop();

    /* Test */
	std::cout << fCPURes << std::endl;

	/* CPU REDUCTION **************************************************/

    //
	// Get Kernel Launch Parameters
	//
	int blockSize = 0,
		gridSize = 0;

	// Block Dimension / Threads per Block
	chCommandLineGet<int>(&blockSize,"t", argc, argv);
	chCommandLineGet<int>(&blockSize,"threads-per-block", argc, argv);
	blockSize = blockSize != 0 ?
			blockSize : DEFAULT_BLOCK_DIM;

	gridSize = ceil((float) numElements /(float) blockSize);


	// Device Memory
	float* d_dataIn = NULL;
	float* d_dataOut = NULL;
	cudaMalloc(&d_dataIn,
			static_cast<size_t>(numElements * sizeof(*d_dataIn)));
	cudaMalloc(&d_dataOut,
			static_cast<size_t>(sizeof(*d_dataOut)));

	if (h_dataIn == NULL || h_dataOut == NULL ||
		d_dataIn == NULL || d_dataOut == NULL)
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - Memory allocation failed" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	//
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	cudaMemcpy(d_dataIn, h_dataIn,
			static_cast<size_t>(numElements * sizeof(*d_dataIn)),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataOut, h_dataOut,
			static_cast<size_t>(sizeof(*d_dataOut)),
			cudaMemcpyHostToDevice);

	memCpyH2DTimer.stop();

	if (blockSize > 1024)
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - The number of threads per block is too big" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);
    int sharedMem = block_dim.x * sizeof(float);

    std::cout << "TEST!" << std::endl;

	kernelTimer.start();

	reduction_KernelNaive<<<grid_dim, block_dim, sharedMem>>>(numElements, d_dataIn, d_dataOut);
	reduction_KernelNaive<<<1, grid_dim, sharedMem>>>(grid_dim.x, d_dataOut, d_dataOut);

    kernelTimer.stop();

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  	<< std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

	//
	// Copy Back Data
	//
	memCpyD2HTimer.start();

	cudaMemcpy(h_dataOut, d_dataOut,
			static_cast<size_t>(sizeof(*d_dataOut)),
			cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

    if(h_dataOut[0] != numElements) {
        std::cout << "Result of reduction is not equal (CPU - GPU)" << std::endl;
        std::cout << h_dataOut[0] << std::endl;
    }
    std::cout << h_dataOut[0] << std::endl;
    //
    //
    //
    /* Optimized */
    *h_dataOut = 0;
    cudaMemcpy(d_dataIn, h_dataIn,
			static_cast<size_t>(numElements * sizeof(*d_dataIn)),
			cudaMemcpyHostToDevice);

    gridSize = ceil((float) numElements /(float) blockSize);
    grid_dim = dim3(gridSize);
	block_dim = dim3(blockSize);
    sharedMem = block_dim.x * sizeof(float);

	kernelOptTimer.start();

	reduction_KernelOptimized<<<grid_dim, block_dim, sharedMem>>>(numElements, d_dataIn, d_dataOut);
	reduction_KernelOptimized<<<1, grid_dim, sharedMem>>>(grid_dim.x, d_dataOut, d_dataOut);

    kernelOptTimer.stop();

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  	<< std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

    memCpyD2HTimer.start();

	cudaMemcpy(h_dataOut, d_dataOut,
			static_cast<size_t>(sizeof(*d_dataOut)),
			cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

    /* Test */
    if(h_dataOut[0] != numElements) {
        std::cout << "Result of reduction is not equal (CPU - GPU Optimized)" << std::endl;
        std::cout << h_dataOut[0] << std::endl;
    }
    std::cout << h_dataOut[0] << std::endl;

	// Free Memory
	if (!pinnedMemory)
	{
		free(h_dataIn);
		free(h_dataOut);
	}
	else
	{
		cudaFreeHost(h_dataIn);
		cudaFreeHost(h_dataOut);
	}
	cudaFree(d_dataIn);
	cudaFree(d_dataOut);

	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    Num Elements            : " << numElements << std::endl
              << "***    Num Threads             : " << blockSize << std::endl
              << "***    Num Block               : " << gridSize << std::endl

			  << "***    Time to Copy to Device  : " << 1e3 * memCpyH2DTimer.getTime()
			  	<< " ms" << std::endl
			  << "***    Copy Bandwidth          : "
			  	<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(*h_dataIn))
			  	<< " GB/s" << std::endl
			  << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
			  	<< " ms" << std::endl
			  << "***    Copy Bandwidth          : "
			  	<< 1e-9 * memCpyD2HTimer.getBandwidth(sizeof(*h_dataOut))
				<< " GB/s" << std::endl
			  << "***    Time for Reduction      : " << 1e3 * kernelTimer.getTime()
				  << " ms" << std::endl
			  << "***" << std::endl
			  << "***    Bandwidth Reduction(CPU): " << 1e-9 * cpuTimer.getBandwidth(numElements * sizeof(*h_dataIn))
                  << " GB/s" << std::endl
              << "***    Time for Reduction (CPU): " << 1e3 * cpuTimer.getTime()
                  << " ms" << std::endl
              << "***    Bandwidth Reduction(GPU): " << 1e-9 * kernelTimer.getBandwidth(numElements * sizeof(*h_dataIn))
                  << " GB/s" << std::endl
              << "***    Time for Reduction (GPU): " << 1e3 * kernelTimer.getTime()
                  << " ms" << std::endl
              << "***    SpeedUp CPU - GPU       : " << cpuTimer.getTime() / kernelTimer.getTime() << std::endl
              << "***" << std::endl
			  << "***    Bandwidth Reduction(OPT): " << 1e-9 * kernelOptTimer.getBandwidth(numElements * sizeof(*h_dataIn))
                  << " GB/s" << std::endl
              << "***    Time for Reduction (OPT): " << 1e3 * kernelOptTimer.getTime()
                  << " ms" << std::endl
              << "***    SpeedUp CPU - OPT       : " << cpuTimer.getTime() / kernelOptTimer.getTime() << std::endl;


	return 0;
}

void
printHelp(char * argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
			  	<< std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "	Use pinned Memory instead of pageable memory" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "	The size of the Matrix" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>"
			  	<< std::endl
			  << "	The number of threads per block" << std::endl
			  << "" << std::endl;
}
