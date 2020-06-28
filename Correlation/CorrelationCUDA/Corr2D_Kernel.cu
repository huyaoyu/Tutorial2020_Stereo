#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

// Debug stuff.
#define SHOW_VARIABLE(x) \
    std::cout << #x" = " << x << std::endl;

// CUDA related constants.
namespace CUDA_PARAMS
{
const int CUDA_MAX_THREADS_PER_BLOCK = 1024;
const int CUDA_THREADS_PER_WARP = 32;   
}

namespace CORR_PARAMS
{
const double CORR_SMALL = 1e-6;
}

// PTA for PackedTensorAccessor
#define PTA_INDEX_TYPE uint32_t

// ========== Device functions. ==========

template <typename scalar_t> 
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t x)
{
    return 1.0 / ( 1.0 + exp(-x) );
}

// ========== Kernel functions. ==========

/*!
 * \param padding The length of the padding. Single side.
 * 
 * This kernel should be launched with block arrangement
 * width coverage * height coverage * baches
 * and thread arrangement
 * x * y ( width, height )
 */
template <typename scalar_t> 
__global__ void k_from_BCHW_2_BHWC_padded(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input, 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    int padding )
{
    const int idxX    = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxY    = blockIdx.y * blockDim.y + threadIdx.y;
    const int strideX = gridDim.x * blockDim.x;
    const int strideY = gridDim.y * blockDim.y;

    const int b = blockIdx.z;

    const int channels = input.size(1);
    const int height   = input.size(2);
    const int width    = input.size(3);

    scalar_t value = 0.0;

    for ( int c = 0; c < channels; c++ )
    {
        for ( int y = idxY; y < height; y += strideY )
        {
            for ( int x = idxX; x < width; x += strideX )
            {
                // Get the data.
                value = input[b][c][y][x];

                // Output the data.
                output[b][y+padding][x+padding][c] = value;
            }
        }
    }
}

/*!
 * The input tensor is already arranged to have its channel being the first dimension.
 * 
 * \param kernelSize The kernel size, whole size. Should be a positive odd number.
 */
template <typename scalar_t>
__global__ void k_kernel_norm( 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    int kernelSize)
{
    const int idxB = blockIdx.z;

    const int inC = input.size(3);

    const int HP  = input.size(1);
    const int WP  = input.size(2);

    // Kernel.
    const int kernelRadius = kernelSize / 2; // kernelSize is assumed to be and odd number.

    // Shared memory.
    extern __shared__ char sharedMemory[];
    scalar_t* sMem = (scalar_t*)sharedMemory;

    // The index for loading data to the shared memory.
    const int smHeight = blockDim.y + kernelRadius*2;
    const int smWidth  = blockDim.x + kernelRadius*2;
    const int smSize   = smHeight * smWidth;

    // Use all the threads in this block linearly.
    const int smStride = blockDim.y * blockDim.x;

    // The upper-left corner of the shared memory region in the input tensor.
    const int smX0 = blockIdx.x * blockDim.x + kernelRadius - kernelRadius;
    const int smY0 = blockIdx.y * blockDim.y + kernelRadius - kernelRadius;

    // Loop over all location in the shared memory.
    for ( int smIdx = threadIdx.y * blockDim.x + threadIdx.x; smIdx < smSize; smIdx += smStride )
    {
        // Clear the value in the shared memory.
        sMem[smIdx] = static_cast<scalar_t>( 0.0 );

        // The index in the input tensor.
        int smY = smIdx / smWidth + smY0;
        int smX = smIdx % smWidth + smX0;

        // Load one channel of input into the shared memory.
        if ( smX < WP && smY < HP )
        {
            scalar_t cData = static_cast<scalar_t>( 0.0 );

            for ( int c = 0; c < inC; c += 1 )
            {
                cData = input[idxB][smY][smX][c];
                sMem[ smIdx ] += cData * cData;
            }
        }
    }

    __syncthreads();

    // The index in the shared memory of the current thread.
    const int smShiftX = threadIdx.x + kernelRadius;
    const int smShiftY = threadIdx.y + kernelRadius;

    // The location in the input tensor of the current thread.
    const int x = blockIdx.x * blockDim.x + threadIdx.x + kernelRadius;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + kernelRadius;

    // Compute the kernel norm.
    if ( x < WP - kernelRadius && y < HP - kernelRadius )
    {
        scalar_t acc = static_cast<scalar_t>( 0.0 );

        for ( int j = -kernelRadius; j <= kernelRadius; ++j )
        {
            for ( int i = -kernelRadius; i <= kernelRadius; ++i )
            {
                acc += sMem[ ( smShiftY + j )*smWidth + smShiftX + i ];
            }
        }

        // Re-use acc.
        acc = std::sqrt( acc );
        acc = acc < CORR_PARAMS::CORR_SMALL ? static_cast<scalar_t>( acc + CORR_PARAMS::CORR_SMALL ) : acc;

        // Save the value to the output tensor.
        output[idxB][0][y][x] = 1.0 / acc;
    }

    __syncthreads();
}

/*!
 * \param padding The padding width, single size. Should be non-negative.
 * \param kernelSize The kernel size, whole size. Should be a positive odd number.
 * \param maxDisplacement The correlation neighborhood along the x (width) direction. Single side. Should be positive.
 * \param strideK The moving stride of the kernel. Positive.
 * \param strideD The moving stride within the neighborhood for correlation. Positive.
 */
template <typename scalar_t>
__global__ void k_corr_2d_forward( 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input0,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input1,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD)
{
    const int idxC    = threadIdx.x;
    const int strideC = blockDim.x;

    const int idxB    = blockIdx.z;
    const int idxXOut = blockIdx.x;
    const int idxYOut = blockIdx.y;

    // const int B   = input0.size(0);
    // const int H   = input0.size(1) - padding * 2;
    // const int W   = input0.size(2) - padding * 2;
    const int W   = input0.size(2);
    const int inC = input0.size(3);

    // Kernel.
    const int kernelRadius = kernelSize / 2; // kernelSize is assumed to be and odd number.
    const int k2 = kernelSize * kernelSize;
    const int nElements = k2 * inC;

    // Output dimensions.
    const int gridRadius = maxDisplacement / strideD;
    const int outC = gridRadius + 1 + gridRadius;

    // Shared memory.
    extern __shared__ char sharedMemory[];
    scalar_t* kernel0     = (scalar_t*)sharedMemory;
    scalar_t* corrResults = kernel0 + nElements;

    // The upper-left corner of the current kernel.
    // Note that, for normal situation, kernelRadius <= padding >= maxDisplacement.
    const int x0 = idxXOut * strideK + gridRadius*strideD - kernelRadius;
    const int y0 = idxYOut * strideK + gridRadius*strideD - kernelRadius;

    // Load the kernel data of input0 into the shared memory.
    for ( int j = 0; j < kernelSize; j++ ) // Height.
    {
        for ( int i = 0; i < kernelSize; i++ ) // Width.
        {
            int chStart = ( j*kernelSize + i ) * inC;
            for ( int c = idxC; c < inC; c += strideC )
            {
                kernel0[ chStart + c ] = input0[idxB][y0+j][x0+i][c];
            }
        }
    }

    __syncthreads();

    for ( int idxOutC = 0; idxOutC < outC; idxOutC++ )
    {
        corrResults[idxC] = 0.0; // Clear the shared memory.

        int y1 = y0;
        int x1 = x0 - gridRadius * strideD + idxOutC * strideD;

        if ( x1 < 0 || x1 + kernelSize > W ) // If gridRadius * strideD >= padding then the first kernelRadius number of x1 will be negative.
        {
            __syncthreads();

            if ( 0 == idxC )
            {
                output[idxB][idxOutC][idxYOut][idxXOut] = static_cast<scalar_t>(0.0);
            }

            continue;
        }

        for ( int j = 0; j < kernelSize; j++ )
        {
            for ( int i = 0; i < kernelSize; i++ )
            {
                int chStart = ( j*kernelSize + i ) * inC;
                for ( int c = idxC; c < inC; c += strideC )
                {
                    corrResults[idxC] += kernel0[ chStart + c ] * input1[idxB][y1+j][x1+i][c];
                }
            }
        }

        __syncthreads();

        if ( 0 == idxC )
        {
            scalar_t kernelSum = 0.0;

            for ( int i = 0; i < blockDim.x; i++ )
            {
                kernelSum += corrResults[i];
            }

            output[idxB][idxOutC][idxYOut][idxXOut] = kernelSum / static_cast<scalar_t>( nElements );
        }
    }

    // // Test sum after load to shared memory.
    // __syncthreads();

    // if ( 0 == idxC )
    // {
    //     scalar_t s = 0.0;

    //     for ( int j = 0; j < kernelSize; j++ ) // Height.
    //     {
    //         for ( int i = 0; i < kernelSize; i++ ) // Width.
    //         {
    //             int chStart = ( j*kernelSize + i ) * inC;
    //             for ( int c = 0; c < inC; c++ )
    //             {
    //                 s += kernel0[ chStart + c ];
    //             }
    //         }
    //     }

    //     output[idxB][0][idxYOut][idxXOut] = s;
    // }
}

template <typename scalar_t> 
__global__ void k_corr_2d_backward_0(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input1,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output0,
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    const int x0 = blockIdx.x + padding;
    const int y0 = blockIdx.y + padding;
    const int idxInC = blockIdx.z;

    const int gradH = grad.size(2);
    const int gradW = grad.size(3);

    const int gridOffset    = threadIdx.x; // The channel index of grad.
    const int gridRadius    = maxDisplacement / strideD;
    const int gridSize      = gridRadius + 1 + gridRadius; // The number of channels of grad.
    const int gridIdxStride = blockDim.x;
    
    const int W = input1.size(2); // Padded.
    const int B = input1.size(0); // Same with output0.

    const int kernelRadius = kernelSize / 2;
    const int nEles = kernelSize * kernelSize * input1.size(3); // Already re-ordered.
    
    // The indices in grad that correspond to the kernels that cover the (x0, y0) position in input0.
    int xGMin = ( x0 - gridRadius*strideD - kernelRadius ) / strideK; // Padded.
    int yGMin = ( y0 - gridRadius*strideD - kernelRadius ) / strideK;

    int xGMax = ( x0 - gridRadius*strideD + kernelRadius ) / strideK;
    int yGMax = ( y0 - gridRadius*strideD + kernelRadius ) / strideK;

    if ( xGMax < 0 || yGMax < 0 || xGMin > gradW - 1 || yGMin > gradH - 1 )
    {
        return;
    }

    // Clipping the indices.
    xGMin = max( 0, xGMin );
    xGMax = min( gradW - 1, xGMax );
    yGMin = max( 0, yGMin );
    yGMax = min( gradH - 1, yGMax );

    extern __shared__ char sum[]; // Should be the number of threads in this block.

    scalar_t* sumG = (scalar_t*)sum;

    for ( int b = 0; b < B; b++ )
    {
        sumG[gridOffset] = 0.0;

        for ( int g = gridOffset; g < gridSize; g += gridIdxStride )
        {
            int y1 = y0;
            int x1 = x0 - gridRadius * strideD + g * strideD; // Padded.

            scalar_t value1 = input1[b][y1][x1][idxInC]; // Input1 is padded.

            for ( int yG = yGMin; yG <= yGMax; yG++ )
            {
                for ( int xG = xGMin; xG <= xGMax; xG++ )
                {
                    // int xL0 = xG * strideK + gridRadius*strideD + kernelRadius; // Padded.
                    int xL0 = xG * strideK + gridRadius*strideD; // Padded.
                    int xL1 = xL0 - gridRadius * strideD + g * strideD;  // Padded.

                    if ( xL1 - kernelRadius < 0 || xL1 + kernelRadius > W - 1 )
                    {
                        continue;
                    }

                    sumG[gridOffset] += grad[b][g][yG][xG] * value1;
                }
            }
        }

        __syncthreads();

        if ( 0 == gridOffset )
        {
            scalar_t acc = 0;
            for ( int g = 0; g < blockDim.x; g++ )
            {
                acc += sumG[g];
            }

            output0[b][idxInC][y0 - padding][x0 - padding] = acc / nEles;
        }

        __syncthreads();
    }
}

template <typename scalar_t> 
__global__ void k_corr_2d_backward_1(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input0,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output1,
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    const int x1 = blockIdx.x + padding;
    const int y1 = blockIdx.y + padding;
    const int idxInC = blockIdx.z;

    const int gradH = grad.size(2);
    const int gradW = grad.size(3);

    const int gridOffset    = threadIdx.x; // The channel index of grad.
    const int gridRadius    = maxDisplacement / strideD;
    const int gridSize      = gridRadius + 1 + gridRadius; // The number of channels of grad.
    const int gridIdxStride = blockDim.x;
    
    const int W = input0.size(2); // Padded.
    const int B = input0.size(0); // Same with output1.

    const int kernelRadius = kernelSize / 2;
    const int nEles = kernelSize * kernelSize * input0.size(3); // Already re-ordered.

    extern __shared__ char sum[]; // Should be the number of threads in this block.

    scalar_t* sumG = (scalar_t*)sum;

    for ( int b = 0; b < B; b++ )
    {
        sumG[gridOffset] = 0.0;

        for ( int g = gridOffset; g < gridSize; g += gridIdxStride )
        {
            int y0 = y1;
            int x0 = x1 + gridRadius * strideD - g * strideD; // Padded.

            // The indices in grad that correspond to the kernels that cover the (x1, y1) position in input1.
            int xGMin = ( x0 - gridRadius*strideD - kernelRadius ) / strideK; // Padded.
            int yGMin = ( y0 - gridRadius*strideD - kernelRadius ) / strideK;

            int xGMax = ( x0 - gridRadius*strideD + kernelRadius ) / strideK;
            int yGMax = ( y0 - gridRadius*strideD + kernelRadius ) / strideK;

            if ( xGMax < 0 || yGMax < 0 || xGMin > gradW - 1 || yGMin > gradH - 1 )
            {
                continue;
            }

            // Clipping the indices.
            xGMin = max( 0, xGMin );
            xGMax = min( gradW - 1, xGMax );
            yGMin = max( 0, yGMin );
            yGMax = min( gradH - 1, yGMax );

            scalar_t value0 = input0[b][y0][x0][idxInC]; // input0 is padded.

            for ( int yG = yGMin; yG <= yGMax; yG++ )
            {
                for ( int xG = xGMin; xG <= xGMax; xG++ )
                {
                    // int xL0 = xG * strideK + gridRadius*strideD + kernelRadius; // Padded.
                    int xL0 = xG * strideK + gridRadius*strideD; // Padded.
                    int xL1 = xL0 - gridRadius * strideD + g * strideD;         // Padded.

                    if ( xL1 - kernelRadius < 0 || xL1 + kernelRadius > W - 1 )
                    {
                        continue;
                    }

                    sumG[gridOffset] += grad[b][g][yG][xG] * value0;
                }
            }
        }

        __syncthreads();

        if ( 0 == gridOffset )
        {
            scalar_t acc = 0;
            for ( int g = 0; g < blockDim.x; g++ )
            {
                acc += sumG[g];
            }

            output1[b][idxInC][y1 - padding][x1 - padding] = acc / nEles;
        }

        __syncthreads();
    }
}

/*!
 * \param padding The padding width, single size. Should be non-negative.
 * \param kernelSize The kernel size, whole size. Should be a positive odd number.
 * \param maxDisplacement The correlation neighborhood along the x (width) direction. Single side. Should be positive.
 * \param strideK The moving stride of the kernel. Positive.
 * \param strideD The moving stride within the neighborhood for correlation. Positive.
 */
template <typename scalar_t>
__global__ void k_corr_2d_forward_zn( 
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input0,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input1,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> L0,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> L1,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD)
{
    const int idxC    = threadIdx.x;
    const int strideC = blockDim.x;

    const int idxB    = blockIdx.z;
    const int idxXOut = blockIdx.x;
    const int idxYOut = blockIdx.y;

    // const int B   = input0.size(0);
    // const int H   = input0.size(1) - padding * 2;
    // const int W   = input0.size(2) - padding * 2;
    const int W   = input0.size(2);
    const int inC = input0.size(3);

    // Kernel.
    const int kernelRadius = kernelSize / 2; // kernelSize is assumed to be and odd number.
    const int k2 = kernelSize * kernelSize;
    const int nElements = k2 * inC;

    // Output dimensions.
    const int gridRadius = maxDisplacement / strideD;
    const int outC = gridRadius + 1 + gridRadius;

    // Shared memory.
    extern __shared__ char sharedMemory[];
    scalar_t* kernel0     = (scalar_t*)sharedMemory;
    scalar_t* corrResults = kernel0 + nElements;

    // The upper-left corner of the current kernel.
    // Note that, for normal situation, kernelRadius <= padding >= maxDisplacement.
    const int x0 = idxXOut * strideK + gridRadius*strideD - kernelRadius;
    const int y0 = idxYOut * strideK + gridRadius*strideD - kernelRadius;

    // Load the kernel data of input0 into the shared memory.
    for ( int j = 0; j < kernelSize; j++ ) // Height.
    {
        for ( int i = 0; i < kernelSize; i++ ) // Width.
        {
            int chStart = ( j*kernelSize + i ) * inC;
            for ( int c = idxC; c < inC; c += strideC )
            {
                kernel0[ chStart + c ] = input0[idxB][y0+j][x0+i][c];
            }
        }
    }

    const scalar_t KL0 = L0[idxB][0][y0 + kernelRadius][x0 + kernelRadius];

    __syncthreads();

    for ( int idxOutC = 0; idxOutC < outC; idxOutC++ )
    {
        corrResults[idxC] = 0.0; // Clear the shared memory.

        int y1 = y0;
        int x1 = x0 - gridRadius * strideD + idxOutC * strideD;

        if ( x1 < 0 || x1 + kernelSize > W ) // If gridRadius * strideD >= padding then the first kernelRadius number of x1 will be negative.
        {
            __syncthreads();

            if ( 0 == idxC )
            {
                output[idxB][idxOutC][idxYOut][idxXOut] = static_cast<scalar_t>(0.0);
            }

            continue;
        }

        for ( int j = 0; j < kernelSize; j++ )
        {
            for ( int i = 0; i < kernelSize; i++ )
            {
                int chStart = ( j*kernelSize + i ) * inC;
                for ( int c = idxC; c < inC; c += strideC )
                {
                    corrResults[idxC] += kernel0[ chStart + c ] * input1[idxB][y1+j][x1+i][c];
                }
            }
        }

        __syncthreads();

        if ( 0 == idxC )
        {
            scalar_t KL1 = L1[idxB][0][y1 + kernelRadius][x1 + kernelRadius];
            scalar_t kernelSum = 0.0;

            for ( int i = 0; i < blockDim.x; i++ )
            {
                kernelSum += corrResults[i];
            }

            // output[idxB][idxOutC][idxYOut][idxXOut] = kernelSum / static_cast<scalar_t>( nElements );
            output[idxB][idxOutC][idxYOut][idxXOut] = kernelSum * KL0 * KL1;
        }
    }

    // // Test sum after load to shared memory.
    // __syncthreads();

    // if ( 0 == idxC )
    // {
    //     scalar_t s = 0.0;

    //     for ( int j = 0; j < kernelSize; j++ ) // Height.
    //     {
    //         for ( int i = 0; i < kernelSize; i++ ) // Width.
    //         {
    //             int chStart = ( j*kernelSize + i ) * inC;
    //             for ( int c = 0; c < inC; c++ )
    //             {
    //                 s += kernel0[ chStart + c ];
    //             }
    //         }
    //     }

    //     output[idxB][0][idxYOut][idxXOut] = s;
    // }
}

template <typename scalar_t> 
__global__ void k_corr_2d_backward_zn_0(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input0,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input1,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cr,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> L0,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> L1,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output0,
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    const int x0 = blockIdx.x + padding;
    const int y0 = blockIdx.y + padding;
    const int idxInC = blockIdx.z;

    const int gradH = grad.size(2);
    const int gradW = grad.size(3);

    const int gridOffset    = threadIdx.x; // The channel index of grad.
    const int gridRadius    = maxDisplacement / strideD;
    const int gridSize      = gridRadius + 1 + gridRadius; // The number of channels of grad.
    const int gridIdxStride = blockDim.x;
    
    const int W = input1.size(2); // Padded.
    const int B = input1.size(0); // Same with output0.

    const int kernelRadius = kernelSize / 2;
    const int nEles = kernelSize * kernelSize * input1.size(3); // Already re-ordered.
    
    // The indices in grad that correspond to the kernels that cover the (x0, y0) position in input0.
    int xGMin = ( x0 - gridRadius*strideD - kernelRadius ) / strideK; // Padded.
    int yGMin = ( y0 - gridRadius*strideD - kernelRadius ) / strideK;

    int xGMax = ( x0 - gridRadius*strideD + kernelRadius ) / strideK;
    int yGMax = ( y0 - gridRadius*strideD + kernelRadius ) / strideK;

    if ( xGMax < 0 || yGMax < 0 || xGMin > gradW - 1 || yGMin > gradH - 1 )
    {
        return;
    }

    // Clipping the indices.
    xGMin = max( 0, xGMin );
    xGMax = min( gradW - 1, xGMax );
    yGMin = max( 0, yGMin );
    yGMax = min( gradH - 1, yGMax );

    extern __shared__ char sum[]; // Should be the number of threads in this block.

    scalar_t* sumG = (scalar_t*)sum;

    for ( int b = 0; b < B; b++ )
    {
        sumG[gridOffset] = 0.0;

        scalar_t value0 = input0[b][y0][x0][idxInC];

        for ( int g = gridOffset; g < gridSize; g += gridIdxStride )
        {
            int y1 = y0;
            int x1 = x0 - gridRadius * strideD + g * strideD; // Padded.

            scalar_t value1 = input1[b][y1][x1][idxInC]; // Input1 is padded.

            for ( int yG = yGMin; yG <= yGMax; yG++ )
            {
                // int yL0 = yG * strideK + kernelRadius;
                int yL0 = yG * strideK + gridRadius*strideD;
                int yL1 = yL0;

                for ( int xG = xGMin; xG <= xGMax; xG++ )
                {
                    // int xL0 = xG * strideK + gridRadius*strideD + kernelRadius; // Padded.
                    int xL0 = xG * strideK + gridRadius*strideD; // Padded.
                    int xL1 = xL0 - gridRadius * strideD + g * strideD;  // Padded.

                    if ( xL1 - kernelRadius < 0 || xL1 + kernelRadius > W - 1 )
                    {
                        continue;
                    }

                    scalar_t crKernel = cr[b][g][yG][xG];
                    scalar_t L0Kernel = L0[b][0][yL0][xL0];
                    scalar_t L1Kernel = L1[b][0][yL1][xL1];

                    // sumG[gridOffset] += grad[b][g][yG][xG] * value1;
                    sumG[gridOffset] += grad[b][g][yG][xG] * 
                        ( -L0Kernel * L0Kernel * value0 * crKernel + L0Kernel * L1Kernel * value1 );
                }
            }
        }

        __syncthreads();

        if ( 0 == gridOffset )
        {
            scalar_t acc = 0;
            for ( int g = 0; g < blockDim.x; g++ )
            {
                acc += sumG[g];
            }

            // output0[b][idxInC][y0 - padding][x0 - padding] = acc / nEles;
            output0[b][idxInC][y0 - padding][x0 - padding] = acc;
        }

        __syncthreads();
    }
}

template <typename scalar_t> 
__global__ void k_corr_2d_backward_zn_1(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input0,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input1,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> cr,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> L0,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> L1,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output1,
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    const int x1 = blockIdx.x + padding;
    const int y1 = blockIdx.y + padding;
    const int idxInC = blockIdx.z;

    const int gradH = grad.size(2);
    const int gradW = grad.size(3);

    const int gridOffset    = threadIdx.x; // The channel index of grad.
    const int gridRadius    = maxDisplacement / strideD;
    const int gridSize      = gridRadius + 1 + gridRadius; // The number of channels of grad.
    const int gridIdxStride = blockDim.x;
    
    const int W = input0.size(2); // Padded.
    const int B = input0.size(0); // Same with output1.

    const int kernelRadius = kernelSize / 2;
    const int nEles = kernelSize * kernelSize * input0.size(3); // Already re-ordered.

    extern __shared__ char sum[]; // Should be the number of threads in this block.

    scalar_t* sumG = (scalar_t*)sum;

    for ( int b = 0; b < B; b++ )
    {
        sumG[gridOffset] = 0.0;

        scalar_t value1 = input1[b][y1][x1][idxInC]; // Padded.

        for ( int g = gridOffset; g < gridSize; g += gridIdxStride )
        {
            int y0 = y1;
            int x0 = x1 + gridRadius * strideD - g * strideD; // Padded.

            // The indices in grad that correspond to the kernels that cover the (x1, y1) position in input1.
            int xGMin = ( x0 - gridRadius*strideD - kernelRadius ) / strideK; // Padded.
            int yGMin = ( y0 - gridRadius*strideD - kernelRadius ) / strideK;

            int xGMax = ( x0 - gridRadius*strideD + kernelRadius ) / strideK;
            int yGMax = ( y0 - gridRadius*strideD + kernelRadius ) / strideK;

            if ( xGMax < 0 || yGMax < 0 || xGMin > gradW - 1 || yGMin > gradH - 1 )
            {
                continue;
            }

            // Clipping the indices.
            xGMin = max( 0, xGMin );
            xGMax = min( gradW - 1, xGMax );
            yGMin = max( 0, yGMin );
            yGMax = min( gradH - 1, yGMax );

            scalar_t value0 = input0[b][y0][x0][idxInC]; // input0 is padded.

            for ( int yG = yGMin; yG <= yGMax; yG++ )
            {
                // int yL0 = yG * strideK + kernelRadius;
                int yL0 = yG * strideK + gridRadius*strideD;
                int yL1 = yL0;

                for ( int xG = xGMin; xG <= xGMax; xG++ )
                {
                    // int xL0 = xG * strideK + gridRadius*strideD + kernelRadius; // Padded.
                    int xL0 = xG * strideK + gridRadius*strideD; // Padded.
                    int xL1 = xL0 - gridRadius * strideD + g * strideD;         // Padded.

                    if ( xL1 - kernelRadius < 0 || xL1 + kernelRadius > W - 1 )
                    {
                        continue;
                    }

                    scalar_t crKernel = cr[b][g][yG][xG];
                    scalar_t L0Kernel = L0[b][0][yL0][xL0];
                    scalar_t L1Kernel = L1[b][0][yL1][xL1];

                    // sumG[gridOffset] += grad[b][g][yG][xG] * value0;
                    sumG[gridOffset] += grad[b][g][yG][xG] * 
                        ( -L1Kernel * L1Kernel * value1 * crKernel + L0Kernel * L1Kernel * value0 );
                }
            }
        }

        __syncthreads();

        if ( 0 == gridOffset )
        {
            scalar_t acc = 0;
            for ( int g = 0; g < blockDim.x; g++ )
            {
                acc += sumG[g];
            }

            // output1[b][idxInC][y1 - padding][x1 - padding] = acc / nEles;
            output1[b][idxInC][y1 - padding][x1 - padding] = acc;
        }

        __syncthreads();
    }
}

// ========== Interface functions. ==========

torch::Tensor from_BCHW_2_BHWC_padded_cuda( torch::Tensor input, int padding )
{
    auto b = input.size(0);
    auto c = input.size(1);
    auto h = input.size(2);
    auto w = input.size(3);

    // Create a padded tensor.
    auto output = torch::zeros({b, h + padding*2, w + padding*2, c}, input.options());

    // Kernel launch specification.
    const int threadsX = 16;
    const int threadsY = 16;
    const dim3 blocks( ( w + threadsX - 1 ) / threadsX, ( h + threadsY - 1 ) / threadsY, b );
    const dim3 thrds( threadsX, threadsY, 1 );

    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( input.scalar_type(), "test_from_BCHW_2_BHWC_padded_cuda", ( [&] {
            k_from_BCHW_2_BHWC_padded<scalar_t><<<blocks, thrds>>>( 
                input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                padding );
        } ) );

    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return output;
}

torch::Tensor create_L(torch::Tensor r, const int kernelSize)
{
    const int B = r.size(0);
    const int H = r.size(1);
    const int W = r.size(2);

    // r is the rearranged tensor with its channel being the first dimension.
    auto L = torch::zeros({B, 1, H, W}, r.options());

    // Kernel.
    const int kernelRadius = kernelSize / 2; // kernelSize is assumed to be an odd number.

    // Kernel launch specification.
    const int threadsX = 16;
    const int threadsY = 16;
    const dim3 blocks( 
        ( W - kernelRadius*2 + threadsX - 1 ) / threadsX, 
        ( H - kernelRadius*2 + threadsY - 1 ) / threadsY, 
        B );
    const dim3 thrds( threadsX, threadsY, 1 );

    // Shared memory.
    // Shared memory should hold data for all the threads of a block and
    // the data bounding this block with the thickness of kernelRadius.
    const int sizeSharedMemory = ( threadsX + kernelRadius*2 ) * ( threadsY + kernelRadius*2 );

    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r.scalar_type(), "create_L", ( [&] {
        k_kernel_norm<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            r.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            L.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            kernelSize );
    } ) );

    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return L;
}

std::vector<torch::Tensor> corr_2d_forward_cuda( 
    torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    // Get the dimensions of the original input.
    const int B = input0.size(0);
    const int H = input0.size(2);
    const int W = input0.size(3);

    const int inC = input0.size(1);

    // kernelSize is assumed to be an odd number.
    // NOTE: For normal situations, kernelRadius == padding.
    // const int kernelRadius = ( kernelSize - 1 ) / 2;

    const int paddedInputH = H + padding*2;
    const int paddedInputW = W + padding*2;

    const int gridRadius = maxDisplacement / strideD;
    const int gridRadDisp = gridRadius * strideD;

    // const auto outH = static_cast<int>( ceil( static_cast<float>(paddedInputH - kernelRadius * 2 - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );
    // const auto outW = static_cast<int>( ceil( static_cast<float>(paddedInputW - kernelRadius * 2 - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );
    const auto outH = static_cast<int>( ceil( static_cast<float>(paddedInputH - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );
    const auto outW = static_cast<int>( ceil( static_cast<float>(paddedInputW - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );

    const int outC = gridRadius + 1 + gridRadius; // The output channels

    // Rearrange the inputs.
    auto r0 = from_BCHW_2_BHWC_padded_cuda(input0, padding);
    auto r1 = from_BCHW_2_BHWC_padded_cuda(input1, padding);

    // Create the output.
    auto output = torch::zeros( { B, outC, outH, outW }, input0.options() );

    // Kernel launch specification.
    const int threads = CUDA_PARAMS::CUDA_THREADS_PER_WARP;
    const dim3 blocks( outW, outH, B );
    const dim3 thrds( threads, 1, 1 );

    // Shared memory size.
    // The size of one kernel across all the input channels, and 
    // additional space for saving the correlation results for
    // each thread in a block.
    const int sizeSharedMemory = 
        kernelSize * kernelSize * inC + threads;

    // CUDA context check.
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r0.scalar_type(), "corr_2d_forward_cuda", ( [&] {
        k_corr_2d_forward<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            r0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            padding, kernelSize, maxDisplacement, strideK, strideD );
    } ) );

    // CUDA context check.
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return { output };
}

std::vector<torch::Tensor> corr_2d_backward_cuda( torch::Tensor grad, torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    // Get the dimensions of the original input.
    const int B = input0.size(0);
    const int H = input0.size(2);
    const int W = input0.size(3);

    const int inC = input0.size(1);

    // Output.
    auto output0 = torch::zeros_like(input0);
    auto output1 = torch::zeros_like(input1);

    // // Rearrange the inputs.
    auto r0 = from_BCHW_2_BHWC_padded_cuda(input0, padding);
    auto r1 = from_BCHW_2_BHWC_padded_cuda(input1, padding);

    // Kernel launch specification.
    // const int threads = CUDA_PARAMS::CUDA_MAX_THREADS_PER_BLOCK;
    const int threads = CUDA_PARAMS::CUDA_THREADS_PER_WARP;
    const dim3 blocks( W, H, inC );
    const dim3 thrds( threads, 1, 1 );

    // Shared memory size.
    // The size of one kernel across all the input channels and 
    // additional space for saving the correlation results for
    // each thread in a block.
    const int sizeSharedMemory = threads;

    // CUDA context check.
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r0.scalar_type(), "corr_2d_backward_cuda_0", ( [&] {
        k_corr_2d_backward_0<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            padding, kernelSize, maxDisplacement, strideK, strideD );
    } ) );

    // CUDA context check.
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r1.scalar_type(), "corr_2d_backward_cuda_1", ( [&] {
        k_corr_2d_backward_1<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            padding, kernelSize, maxDisplacement, strideK, strideD );
    } ) );

    // CUDA context check.
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return { output0, output1 };
}

std::vector<torch::Tensor> corr_2d_forward_zn_cuda( 
    torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    // Get the dimensions of the original input.
    const int B = input0.size(0);
    const int H = input0.size(2);
    const int W = input0.size(3);

    const int inC = input0.size(1);

    // kernelSize is assumed to be an odd number.
    // NOTE: For normal situations, kernelRadius == padding.
    // const int kernelRadius = ( kernelSize - 1 ) / 2;

    const int paddedInputH = H + padding*2;
    const int paddedInputW = W + padding*2;

    const int gridRadius = maxDisplacement / strideD;
    const int gridRadDisp = gridRadius * strideD;

    // const auto outH = static_cast<int>( ceil( static_cast<float>(paddedInputH - kernelRadius * 2 - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );
    // const auto outW = static_cast<int>( ceil( static_cast<float>(paddedInputW - kernelRadius * 2 - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );
    const auto outH = static_cast<int>( ceil( static_cast<float>(paddedInputH - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );
    const auto outW = static_cast<int>( ceil( static_cast<float>(paddedInputW - 2 * gridRadius*strideD) / static_cast<float>(strideK) ) );

    const int outC = gridRadius + 1 + gridRadius; // The output channels

    // Rearrange the inputs.
    auto r0 = from_BCHW_2_BHWC_padded_cuda(input0, padding);
    auto r1 = from_BCHW_2_BHWC_padded_cuda(input1, padding);

    // Create two new tensors for the L values.
    auto L0 = create_L(r0, kernelSize);
    auto L1 = create_L(r1, kernelSize);

    // // Debug.
    // SHOW_VARIABLE(B);
    // SHOW_VARIABLE(H);
    // SHOW_VARIABLE(W);
    // SHOW_VARIABLE(inC);
    // SHOW_VARIABLE(outH);
    // SHOW_VARIABLE(outW);
    // SHOW_VARIABLE(gridRadius);
    // SHOW_VARIABLE(r0.size(0));
    // SHOW_VARIABLE(r0.size(1));
    // SHOW_VARIABLE(r0.size(2));
    // SHOW_VARIABLE(r0.size(3));
    // SHOW_VARIABLE(r1.size(0));
    // SHOW_VARIABLE(r1.size(1));
    // SHOW_VARIABLE(r1.size(2));
    // SHOW_VARIABLE(r1.size(3));

    // Create the output.
    auto output = torch::zeros( { B, outC, outH, outW }, input0.options() );

    // Kernel launch specification.
    const int threads = CUDA_PARAMS::CUDA_THREADS_PER_WARP;
    const dim3 blocks( outW, outH, B );
    const dim3 thrds( threads, 1, 1 );

    // Shared memory size.
    // The size of one kernel across all the input channels, and 
    // additional space for saving the correlation results for
    // each thread in a block.
    const int sizeSharedMemory = 
        kernelSize * kernelSize * inC + threads;

    // CUDA context check.
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r0.scalar_type(), "corr_2d_forward_zn_cuda", ( [&] {
        k_corr_2d_forward_zn<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            r0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            L0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            L1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            padding, kernelSize, maxDisplacement, strideK, strideD );
    } ) );

    // CUDA context check.
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return { output, L0, L1 };
}

std::vector<torch::Tensor> corr_2d_backward_zn_cuda( torch::Tensor grad, torch::Tensor input0, torch::Tensor input1, 
    torch::Tensor cr, torch::Tensor L0, torch::Tensor L1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    // Get the dimensions of the original input.
    const int B = input0.size(0);
    const int H = input0.size(2);
    const int W = input0.size(3);

    const int inC = input0.size(1);

    // Output.
    auto output0 = torch::zeros_like(input0);
    auto output1 = torch::zeros_like(input1);

    // // Rearrange the inputs.
    auto r0 = from_BCHW_2_BHWC_padded_cuda(input0, padding);
    auto r1 = from_BCHW_2_BHWC_padded_cuda(input1, padding);

    // Kernel launch specification.
    // const int threads = CUDA_PARAMS::CUDA_MAX_THREADS_PER_BLOCK;
    const int threads = CUDA_PARAMS::CUDA_THREADS_PER_WARP;
    const dim3 blocks( W, H, inC );
    const dim3 thrds( threads, 1, 1 );

    // Shared memory size.
    // The size of one kernel across all the input channels and 
    // additional space for saving the correlation results for
    // each thread in a block.
    const int sizeSharedMemory = threads;

    // CUDA context check.
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r0.scalar_type(), "corr_2d_backward_zn_cuda_0", ( [&] {
        k_corr_2d_backward_zn_0<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            cr.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            L0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            L1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            padding, kernelSize, maxDisplacement, strideK, strideD );
    } ) );

    // CUDA context check.
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    // Kernel launch.
    AT_DISPATCH_FLOATING_TYPES( r1.scalar_type(), "corr_2d_backward_cuda_zn_1", ( [&] {
        k_corr_2d_backward_zn_1<scalar_t><<<blocks, thrds, sizeSharedMemory*sizeof(scalar_t)>>>( 
            grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            r1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            cr.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            L0.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            L1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output1.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            padding, kernelSize, maxDisplacement, strideK, strideD );
    } ) );

    // CUDA context check.
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        std::stringstream ss;
        ss << __FILE__ << ": "<< __LINE__ << ": cudaGetLastError() returns " << err;
        throw std::runtime_error(ss.str());
    }

    return { output0, output1 };
}
