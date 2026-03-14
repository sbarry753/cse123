# Matrix Multiplication Benchmarking

## Overview
When benchmarking audio effects, the effects themselves did not introduce substantial computational load on the CPU. To better understand the limits of the Daisy Seed's MCU, we decided it was viable to stress-test the MCU with matrix multiplication. Matrix multiplication is a very computationally demanding operation, stressing arithmetic throughput and memory bandwidth. It is an $O(n^3)$ operation, scaling poorly as $n$ (largest matrix dimension) increases. Additionally, matrix multiplication is a particularly suitable experiment to run in since we plan to run neural network inference on the MCU. Neural networks compute many matrix multiplications in a single forward pass, so it is imperative we know matrix multiplication throughput. 

As outlined in the effects benchmarking, the audio callback block size is proportional to the maximum time available in the audio callback. As block size increases, the amount of time we can spend in the callback also increases. A standard method to estimate throughput is by calculating the amount of floating point operations per block (FLOPs/block). In matrix multiplication, for each element of the resulting matrix of height and width *N*, the dot product is computed between a length *N* row vector and length *N* column vector. This results in $2N^3$ FLOPs (multiply + add is two FLOPs) per matrix multiplication.

## Experiment Strategy
Three matrix classes are implemented to not only stress-test the MCU, but also to discover potential optimizations specific to the MCU. The first class stores the data of the matrix as a two-dimensional array. The second class stores the data as a contiguous 1D array. Finally, the third class is a thin wrapper over the *CMSIS-DSP arm_matrix_instance_f32* struct. 

The first class implements the standard mat mul algorithm, a better cache-optimized design, and a tiled approach. The second class implements the same algorithms as the first, but also includes two row pointer caching (simpler inner loop from reloacted address calculations) algorithms. The first row pointer caching algorithm iterates over indices *i-k-j* (cache-aware ordering), while the second one iterates over *i-j-k* (standard ordering). Finally, the third class implements its mat mul as a thin wrapper over the optimized *CMSIS-DSP arm_mat_mult_f32()* function. 

In the audio callback, a mat mul algorithm is repeatedly called. Intially, the algorithm is only run once, but the amount of iterations is increased if the measured CPU load is less than 100%. Once the CPU load is in the range of 90%-100%, CPU load metrics cannot be printed as the matrix multiplications take up virtually all of the available processing time in the callback. When this occurs, the amount of repeated mat muls is at the maximum for the particular block size and is logged in a spreadsheet. From here, the maximum *kFLOPs/block* is calculated with this equation:

$$
kFLOPs/block_{max} = \frac{num\_repeats*(2 * N^3)}{1000}
$$

where *N* is 16, the length of the rows and columns of the three matrices. Note that 16 was chosen because 32 was too large of a matrix for the MCU to finish computation in ~0.167 ms (8 block size). 

For a given algorithm and block size combination, *kFLOPs/block* tells us the maximum FLOPs per callback time. If we need a process to run in a specific time frame, this information gives a a rough estimate of the computational throughput needed to achieve this goal. For example, these calculations can come in handy when we are optimizing neural network inference to run within a specific time budget, say about 5 ms. A block size of 256 runs in ~5.3 ms, so the maximum inference FLOPs should be similar to the experiments with a 256 block size. 

The code is compiled with -O3, -DARM_MATH_CM7, -DUSE_ARM_MATH, and -DARM_MATH_LOOPUNROLL compiler flags.

## Results
[Results Spreadsheet](https://docs.google.com/spreadsheets/d/1QgfL2nDbECkWdAaCYBFUneOOQhCu9Kp85AZZbPZRD8g/edit?usp=sharing)

Interestingly, known matrix optimizations for more "mainstream "(laptops and desktops) CPUs result in worse performance. The *Cache Locality* algorithms (2D and 1D implementations) often result in peformance increases for larger matrix sizes and consumer-grade CPUs. It reorders the triple-loop indices, going from *i-j-k* to *i-k-j*. This provides better cache locality for the second matrix involved in this operation (contiguous access in the inner loop for matrix b). For larger matrices, iteration across different rows of a matrix usually means frequent cache misses and reorder solves this problem. However, since the matrix size is very small (16), elements from different rows may already be present in the cache. Furthermore, the *i-j-k* version accumulates into a local sum (in a register) once per element of the resulting matrix and writes directly to it once. However, the *i-k-j* version must update each element of the resulting matrix per inner loop iteration, resulting in a read-modify-write 16 times per element.

The CMSIS-DSP implementation was clearly the fastest and most performant implementation. If code size is not an issue, the CMSIS-DSP implementations should be used if performance-critical code is necessary. 