/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : main.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <stdio.h>
#include <assert.h>

const static int DEFAULT_MEM_SIZE       = 10*1024*1024; // 10 MB
const static int DEFAULT_NUM_ITERATIONS =         1000;
const static int DEFAULT_BLOCK_DIM      =          128;
const static int DEFAULT_GRID_DIM       =           16;

//
// Function Prototypes
//
void printHelp(char *);

//
// Test Kernel
//

__global__ void 
globalMem2SharedMem(float * d_memoryA, int iSize)
{
  /* Amount of shared memory is determined by host call */
  extern __shared__ float s_memoryA[];
  /* Generate global index */
  int iID = blockDim.x * blockIdx.x + threadIdx.x;
  /* Get the number of available threads */
  int iNumThreads = blockDim.x * gridDim.x;
  /* Calculate number of elements */
  int iNumElements = iSize / sizeof(float);
  /* Read global memory (coalesce) to shared memory */
  /* Avoid bank conflicts */
  for(int i = iID; i < iNumElements; i += iNumThreads)
    s_memoryA[i] = d_memoryA[i];
}

__global__ void 
SharedMem2globalMem(float * d_memoryA, int iSize)
{
  /* Amount of shared memory is determined by host call */
  extern __shared__ float s_memoryA[];
  /* Generate global index */
  int iID = blockDim.x * blockIdx.x + threadIdx.x;
  /* Get the number of available threads */
  int iNumThreads = blockDim.x * gridDim.x;
  /* Calculate number of elements */
  int iNumElements = iSize / sizeof(float);
  /* Read global memory (coalesce) to shared memory */
  for(int i = iID; i < iNumElements; i += iNumThreads)
    d_memoryA[i] = s_memoryA[i];
}

__global__ void 
SharedMem2Registers(float * outFloat, int iSize)
{
  /* Amount of shared memory is determined by host call */
  extern __shared__ float s_memoryA[];
  /* Variable in register */
  float r_var;
  /* Generate global index */
  int iID = blockDim.x * blockIdx.x + threadIdx.x;
  /* Get the number of available threads */
  int iNumThreads = blockDim.x * gridDim.x;
  /* Calculate number of elements */
  int iNumElements = iSize / sizeof(float);
  /* Read global memory (coalesce) to shared memory */
  for(int i = iID; i < iNumElements; i += iNumThreads)
    r_var = s_memoryA[i];
  /* Conditionally assign register var, so it won't get optimized */
  if(iID == 0) outFloat[0] = r_var;
}

__global__ void 
Registers2SharedMem(float * outFloat, int iSize)
{
  /* Amount of shared memory is determined by host call */
  extern __shared__ float s_memoryA[];
  /* Variable in register */
  float r_var;
  /* Generate global index */
  int iID = blockDim.x * blockIdx.x + threadIdx.x;
  /* Get the number of available threads */
  int iNumThreads = blockDim.x * gridDim.x;
  /* Calculate number of elements */
  int iNumElements = iSize / sizeof(float);
  /* Read global memory (coalesce) to shared memory */
  for(int i = iID; i < iNumElements; i += iNumThreads)
    s_memoryA[i] = r_var;
  /* Conditionally assign register var, so it won't get optimized */
  if(iID == 0) outFloat[0] = r_var;
}

__global__ void 
bankConflictsRead(float * outFloat, int iStride, unsigned long long *ullTime)
{
  /* Static size of shared memory; The size doesn't matters */
  __shared__ float s_memoryA[1024];
  /* Variable in register */
  float r_var;
  /* Start measure clock cycles */
  unsigned long long startTime = clock64();
  /* Access data from shared memory to register */
  r_var = s_memoryA[threadIdx.x*iStride];
  /* End measure clock cycles */
  *ullTime = clock64() - startTime;
  /* Conditionally assign register var, so it won't get optimized */
  if(threadId.x == 0) outFloat[0] = r_var;
}

//
// Main
//
int
main ( int argc, char * argv[] )
{
	// Show Help
	bool optShowHelp = chCommandLineGetBool("h", argc, argv);
	if ( !optShowHelp )
		optShowHelp = chCommandLineGetBool("help", argc, argv);

	if ( optShowHelp ) {
		printHelp ( argv[0] );
		exit (0);
	}

	std::cout << "***" << std::endl
	          << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer kernelTimer;

	//
	// Get kernel launch parameters and configuration
	//
	int optNumIterations = 0,
		optBlockSize = 0,
		optGridSize = 0;

	// Number of Iterations
	chCommandLineGet<int> ( &optNumIterations,"i", argc, argv );
	chCommandLineGet<int> ( &optNumIterations,"iterations", argc, argv );
	optNumIterations = ( optNumIterations != 0 ) ? optNumIterations : DEFAULT_NUM_ITERATIONS;

	// Block Dimension / Threads per Block
	chCommandLineGet <int> ( &optBlockSize,"t", argc, argv );
	chCommandLineGet <int> ( &optBlockSize,"threads-per-block", argc, argv );
	optBlockSize = optBlockSize != 0 ? optBlockSize : DEFAULT_BLOCK_DIM;

	if ( optBlockSize > 1024 ) {
		std::cout << "\033[31m***" << std::cout
		          << "*** Error - The number of threads per block is too big" 
				  	<< std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	// Grid Dimension
	chCommandLineGet <int> ( &optGridSize,"g", argc, argv );
	chCommandLineGet <int> ( &optGridSize,"grid-dim", argc, argv );
	optGridSize = optGridSize != 0 ? optGridSize : DEFAULT_GRID_DIM;
	
	dim3 grid_dim = dim3 ( optGridSize );
	dim3 block_dim = dim3 ( optBlockSize );
	
	
	int optModulo = 32*1024; // modulo in access pattern for conflict test
	chCommandLineGet <int> ( &optModulo,"mod", argc, argv );

	int optStride = 1; // modulo in access pattern for conflict test
	chCommandLineGet <int> ( &optStride,"stride", argc, argv );

	// Memory size
	int optMemorySize = 0;
	
	chCommandLineGet <int> ( &optMemorySize, "s", argc, argv );
	chCommandLineGet <int> ( &optMemorySize, "size", argc, argv );
	optMemorySize = optMemorySize != 0 ? optMemorySize : DEFAULT_MEM_SIZE;
	

	//
	// Device Memory
	//
	float* d_memoryA = NULL;
	cudaMalloc ( &d_memoryA, static_cast <size_t> ( optMemorySize ) ); // optMemorySize is in bytes

	float *outFloat = NULL;  // dummy variable to prevent compiler optimizations
	cudaMalloc ( &outFloat, static_cast <float> ( sizeof ( float ) ) );
		
	long hClocks = 0;
	long *dClocks = NULL;
	cudaMalloc ( &dClocks, sizeof ( long ) );
	
	if ( d_memoryA == NULL || dClocks == NULL )
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - Memory allocation failed" << std::endl
		          << "***\033[0m" << std::endl;

		exit (-1);
	}
	
	int shared_dim = optMemorySize;
	unsigned long long ullTime;
	//
	// Tests
	//
	std::cout << "Starting kernel: " << grid_dim.x << "x" << block_dim.x << " threads, " << optMemorySize << "B shared memory" << ", " << optNumIterations << " iterations" << std::endl;
	kernelTimer.start();
	for ( int i = 0; i < optNumIterations; i++ )
	{
		//
		// Launch Kernel
		//
		if ( chCommandLineGetBool ( "global2shared", argc, argv ) )
		{
			globalMem2SharedMem <<< grid_dim, block_dim, shared_dim>>>
					(d_memoryA, optMemorySize);
		}
		else if ( chCommandLineGetBool ( "shared2global", argc, argv ) )
		{
			SharedMem2globalMem <<< grid_dim, block_dim, shared_dim>>>
					(d_memoryA, optMemorySize);
		}
		else if ( chCommandLineGetBool ( "shared2register", argc, argv ) )
		{
			SharedMem2Registers <<< grid_dim, block_dim, shared_dim>>>
					(outFloat, optMemorySize);
		}
		else if ( chCommandLineGetBool ( "register2shared", argc, argv ) )
		{
			Registers2SharedMem <<< grid_dim, block_dim,  shared_dim>>>
					(outFloat, optMemorySize);
		}
		else if ( chCommandLineGetBool ( "shared2register_conflict", argc, argv ) )
		{
			bankConflictsRead <<< 1, 32 >>>
					(outFloat, optStride, &ullTime);
		}
	}

	// Mandatory synchronize after all kernel launches
	cudaDeviceSynchronize();
	kernelTimer.stop();

	cudaError_t cudaError = cudaGetLastError();
	if ( cudaError != cudaSuccess )
	{
		std::cout << "\033[31m***" << std::endl
	    	      << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  << std::endl
		          << "***\033[0m" << std::endl;

		return -1;
	}

	// Print Measurement Results

	if ( chCommandLineGetBool ( "global2shared", argc, argv ) ) {
		std::cout << "Copy global->shared, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_dim.x << ", bDim=" << std::setw(5) << block_dim.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_dim.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
	}
	
	if ( chCommandLineGetBool ( "shared2global", argc, argv ) ) {
		std::cout << "Copy shared->global, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_dim.x << ", bDim=" << std::setw(5) << block_dim.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_dim.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
	}

	if ( chCommandLineGetBool ( "shared2register", argc, argv ) ) {
		std::cout << "Copy shared->register, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_dim.x << ", bDim=" << std::setw(5) << block_dim.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_dim.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
	}
	
	if ( chCommandLineGetBool ( "register2shared", argc, argv ) ) {
		std::cout << "Copy register->shared, size=" << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_dim.x << ", bDim=" << std::setw(5) << block_dim.x;
		//std::cout << ", time=" << kernelTimer.getTime(optNumIterations) << 
		std::cout.precision ( 2 );
		std::cout << ", bw=" << std::fixed << std::setw(6) << ( optMemorySize * grid_dim.x ) / kernelTimer.getTime(optNumIterations) / (1E09) << "GB/s" << std::endl;
	}	

	if ( chCommandLineGetBool ( "shared2register_conflict", argc, argv ) ) {
		if ( chCommandLineGetBool ( "shared2register_conflict", argc, argv ) ) {
			cudaError_t error = cudaMemcpy ( &hClocks, dClocks, sizeof ( long ), cudaMemcpyDeviceToHost );
			if ( error != cudaSuccess) {
				fprintf ( stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString ( error ) );
				return 1;
			}
		}	

		std::cout << "Shared memory bank conflict test, size=1024, gDim=1, bDim=32"; // << std::setw(10) << optMemorySize << ", gDim=" << std::setw(5) << grid_dim.x << ", bDim=" << std::setw(5) << block_dim.x;
		std::cout << ", stride=" << std::setw(6) << optStride << ", modulo=" << std::setw(6) << optModulo;
		std::cout << ", clocks=" << std::setw(10) << ullTime << std::endl; //hClocks << std::endl;
	}
	
	return 0;
}

void
printHelp(char * programName)
{
	std::cout 	
		<< "Usage: " << std::endl
		<< "  " << programName << " [-p] [-s <memory_size>] [-i <num_iterations>]" << std::endl
		<< "                [-t <threads_per_block>] [-g <blocks_per_grid]" << std::endl
		<< "                [-stride <stride>] [-offset <offset>]" << std::endl
		<< "  --{global2shared,shared2global,shared2register,register2shared,shared2register_conflict}" << std::endl
		<< "    Run kernel analyzing shared memory performance" << std::endl
		<< "  -s <memory_size>|--size <memory_size>" << std::endl
		<< "    The amount of memory to allcate" << std::endl
		<< "  -t <threads_per_block>|--threads-per-block <threads_per_block>" << std::endl
		<< "    The number of threads per block" << std::endl
		<< "  -g <blocks_per_grid>|--grid-dim <blocks_per_grid>" << std::endl
		<< "     The number of blocks per grid" << std::endl
		<< "  -i <num_iterations>|--iterations <num_iterations>" << std::endl
		<< "     The number of iterations to launch the kernel" << std::endl
		<< "  -stride <stride>" << std::endl
		<< "     Stride parameter for global-stride test. Not that size parameter is ignored then." << std::endl
		<< "  -offset <offset>" << std::endl
		<< "     Offset parameter for global-offset test. Not that size parameter is ignored then." << std::endl
		<< "" << std::endl;
}
