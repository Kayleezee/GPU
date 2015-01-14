/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 09
 *
 *                           Group : TODO: TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Stencil Code
 *
 *************************************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>

#define _OPT_KERNEL
const static int DEFAULT_NUM_ELEMENTS   = 1024;
const static int DEFAULT_NUM_ITERATIONS =    5;
const static int DEFAULT_BLOCK_DIM      =  128;
const static int DEFAULT_STEPS          =  100;

//
// Structures
struct StencilArray_t {
    float* array;
    int    size; // size == width == height
};

//
// Function Prototypes
//
void printHelp(char *);

//
// Stencil Code Kernel for the speed calculation
//
__device__ bool
isBorder(int iSize, int iIndex)
{
    return  (iIndex / iSize == 0) ||
            (iIndex % iSize == 0) ||
            (iIndex / iSize == iSize - 1) ||
            (iIndex % iSize == iSize - 1) ||
            iIndex > iSize * iSize - 1;
}


__global__ void
simpleStencil_Kernel(StencilArray_t d_array_old, StencilArray_t d_array_new)
{
    // index of array
    int iIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // value of actual element
    float fThisValue = d_array_old.array[iIndex];

    float act_sum;

    for(int i=0; i<DEFAULT_STEPS; i++){
      if(!isBorder(d_array_old.size, iIndex)) {

        act_sum = fThisValue + 24.0 / 100.0
                  * ((-4.0) * fThisValue
                  + d_array_old.array[iIndex + 1]
                  + d_array_old.array[iIndex - 1]
                  + d_array_old.array[iIndex + d_array_old.size]
                  + d_array_old.array[iIndex - d_array_old.size]);
      }
    }
    d_array_new.array[iIndex] = act_sum;
    //__syncthreads();
    //d_array.array[iIndex] = act_sum;
}

__global__ void
optStencil_Kernel(StencilArray_t d_array_old, StencilArray_t d_array_new)
{
    extern __shared__ float shMemBlock[];

    int iIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int size = d_array_old.size;
    int iOffset = size;

    float fThisValue = d_array_old.array[iIndex];

    shMemBlock[threadIdx.x+iOffset] = fThisValue;

    float act_sum;

    for(int i=0; i<DEFAULT_STEPS; i++){
      /* Copy the helo-Values to shared memory */
      if(threadIdx.x < size && iIndex >= size)
        shMemBlock[threadIdx.x] = d_array_old.array[iIndex-size];
      if((threadIdx.x >= blockDim.x - size) && (iIndex < (blockDim.x*gridDim.x - size)))
        shMemBlock[threadIdx.x+2*iOffset] = d_array_old.array[iIndex+size];

      __syncthreads();

      if(!isBorder(size, iIndex)) {

        act_sum = fThisValue + 24.0 / 100.0
                  * ((-4.0) * fThisValue
                  + shMemBlock[threadIdx.x + iOffset + 1]
                  + shMemBlock[threadIdx.x + iOffset - 1]
                  + shMemBlock[threadIdx.x + iOffset + size]
                  + shMemBlock[threadIdx.x + iOffset - size]);
      }

      /* Copy back the helo-Values to global memory */
      if(threadIdx.x < size && iIndex >= size)
        d_array_old.array[iIndex-size] = shMemBlock[threadIdx.x];
      if((threadIdx.x >= blockDim.x - size) && (iIndex < (blockDim.x*gridDim.x - size)))
        d_array_old.array[iIndex+size] = shMemBlock[threadIdx.x+2*iOffset];
    }
    d_array_new.array[iIndex] = act_sum;
}
//
// Main
//
int
main(int argc, char * argv[])
{
    bool showHelp = chCommandLineGetBool("h", argc, argv);
    if (!showHelp) {
        showHelp = chCommandLineGetBool("help", argc, argv);
    }

    if (showHelp) {
        printHelp(argv[0]);
        exit(0);
    }

    std::cout << "***" << std::endl
              << "*** Starting ..." << std::endl
              << "***" << std::endl;

    ChTimer memCpyH2DTimer, memCpyD2HTimer;
    ChTimer kernelTimer, kernelTimer_simple;

    /*
    ofstream fout;
    fout.open("results_2dheat.txt", fstream::app);
    */

    //
    // Allocate Memory
    //
    int numElements = 0;
    chCommandLineGet<int>(&numElements, "s", argc, argv);
    chCommandLineGet<int>(&numElements, "size", argc, argv);
    numElements = numElements != 0 ? numElements : DEFAULT_NUM_ELEMENTS;

    //
    // Host Memory
    //
    bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
    if (!pinnedMemory) {
        pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
    }

    StencilArray_t h_array;
    h_array.size = sqrt(numElements);
    if (!pinnedMemory) {
        // Pageable
        h_array.array = static_cast<float*>
                (malloc(static_cast<size_t>
                (numElements * sizeof(*(h_array.array)))));
    } else {
        // Pinned<F4>
        cudaMallocHost(&(h_array.array),
                static_cast<size_t>
                (numElements * sizeof(*(h_array.array))));
    }

    std::cout << "Step 1" << std::endl;

    // Init Particles
//  srand(static_cast<unsigned>(time(0)));
    srand(0); // Always the same random numbers
    for (int i = 0; i < h_array.size * h_array.size; i++) {
        h_array.array[i] = 0;
        // TODO: Initialize the array
        if (i >= (h_array.size * 0.25) && i <= (h_array.size * 0.75)) {
            h_array.array[i] = 127.0;
        }
    }

    std::cout << "Step 2" << std::endl;

    // Device Memory
    int iIndex = 0;
    StencilArray_t d_array[2];
    cudaMalloc(&(d_array[0].array),
            static_cast<size_t>(numElements * sizeof(*h_array.array)));

    cudaMalloc(&(d_array[1].array),
            static_cast<size_t>(numElements * sizeof(*h_array.array)));

    if (h_array.array == NULL || d_array[0].array == NULL || d_array[1].array == NULL) {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - Memory allocation failed" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }

    std::cout << "Step 3" << std::endl;

    //
    // Copy Data to the Device
    //
    memCpyH2DTimer.start();

    cudaMemcpy(d_array[iIndex].array, h_array.array,
            static_cast<size_t>(numElements * sizeof(*d_array[iIndex].array)),
            cudaMemcpyHostToDevice);

    d_array[0].size = h_array.size;
    d_array[1].size = h_array.size;

    memCpyH2DTimer.stop();

    std::cout << "Step 4" << std::endl;
    //
    // Get Kernel Launch Parameters
    //
    int blockSize = 0,
        gridSize = 0,
        numIterations = 0;

    // Number of Iterations
    chCommandLineGet<int>(&numIterations,"i", argc, argv);
    chCommandLineGet<int>(&numIterations,"num-iterations", argc, argv);
    numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

    // Block Dimension / Threads per Block
    chCommandLineGet<int>(&blockSize,"t", argc, argv);
    chCommandLineGet<int>(&blockSize,"threads-per-block", argc, argv);
    blockSize = blockSize != 0 ? blockSize : DEFAULT_BLOCK_DIM;

    if (blockSize > 1024) {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - The number of threads per block is too big" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }

    //gridSize = ceil(static_cast<float>(d_array[iIndex].size) / static_cast<float>(blockSize));
    gridSize = ceil(numElements / blockSize)+1;

    dim3 grid_dim = dim3(gridSize);
    dim3 block_dim = dim3(blockSize);

    std::cout << "***" << std::endl;
    std::cout << "*** Grid: " << gridSize << std::endl;
    std::cout << "*** Block: " << blockSize << std::endl;
    std::cout << "***" << std::endl;

    float shMemSize = (blockSize + 2 * d_array[iIndex].size) * sizeof(float);

    kernelTimer.start();

    for (int i = 0; i < numIterations; i ++) {
      optStencil_Kernel<<<grid_dim, block_dim, shMemSize>>>(d_array[iIndex], d_array[1-iIndex]);
      cudaDeviceSynchronize();
      iIndex = 1 - iIndex;
    }

    // Synchronize
    //cudaDeviceSynchronize();

    // Check for Errors
    cudaError_t cudaError = cudaGetLastError();
    if ( cudaError != cudaSuccess ) {
        std::cout << "\033[31m***" << std::endl
                  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                  << std::endl
                  << "***\033[0m" << std::endl;

        return -1;
    }

    kernelTimer.stop();

    kernelTimer_simple.start();

    for (int i = 0; i < numIterations; i ++) {
      simpleStencil_Kernel<<<grid_dim, block_dim>>>(d_array[iIndex], d_array[1-iIndex]);
      cudaDeviceSynchronize();
      iIndex = 1 - iIndex;
    }

    // Synchronize
    //cudaDeviceSynchronize();

    // Check for Errors
    cudaError_t cudaError2 = cudaGetLastError();
    if ( cudaError2 != cudaSuccess ) {
        std::cout << "\033[31m***" << std::endl
                  << "***ERROR*** " << cudaError2 << " - " << cudaGetErrorString(cudaError2)
                  << std::endl
                  << "***\033[0m" << std::endl;

        return -1;
    }

    kernelTimer_simple.stop();

    //
    // Copy Back Data
    //
    memCpyD2HTimer.start();

    cudaMemcpy(h_array.array, d_array[1-iIndex].array,
            static_cast<size_t>(h_array.size * sizeof(*(h_array.array))),
            cudaMemcpyDeviceToHost);

    memCpyD2HTimer.stop();

    /*
    for(int i = 0; i < h_array.size * h_array.size; i++) {
            std::cout << h_array.array[i] << "\t";
    }
    */

    // Free Memory
    if (!pinnedMemory) {
        free(h_array.array);
    } else {
        cudaFreeHost(h_array.array);
    }

    cudaFree(d_array[0].array);
    cudaFree(d_array[1].array);

    // Print Meassurement Results
    std::cout << "***" << std::endl
              << "*** Results:" << std::endl
              << "*** You made it. The GPU ran without an error!" << std::endl << std::endl
              << "***    Size: " << numElements << std::endl
              << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: "
                << 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(*h_array.array))
                << " GB/s" << std::endl
              << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: "
                << 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(*h_array.array))
                << " GB/s" << std::endl
              << "***    Time for Stencil Computation (optimized): " << 1e3 * kernelTimer.getTime() / numIterations
                << " ms" << std::endl
              << "***    Time for Stencil Computation (simple): " << 1e3 * kernelTimer_simple.getTime() / numIterations
                << " ms" << std::endl
              << "***" << std::endl;

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
              << "    Use pinned Memory instead of pageable memory" << std::endl
              << "" << std::endl
              << "  -s <width-and-height>|--size <width-and-height>" << std::endl
              << "    THe width and the height of the array" << std::endl
              << "" << std::endl
              << "  -t <threads_per_block>|--threads-per-block <threads_per_block>"
                  << std::endl
              << "    The number of threads per block" << std::endl
              << "" << std::endl;
}
