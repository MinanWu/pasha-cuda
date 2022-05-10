#include <cuda.h>
#include <iostream>

using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

#define NUM_THREADS 256


__constant__ unsigned_int d_vertexExp;

double prob_gpu;
double alpha_gpu;
double delta_gpu;
double epsilon_gpu;


unsigned_int edgeNum_gpu;
unsigned_int edgeCount_gpu;
unsigned_int total_gpu;
unsigned_int size_gpu;
unsigned_int array_slot;

unsigned_int hittingCountStage;
double pathCountStage;

byte* pick_gpu;
byte* edgeArray_gpu;
byte* stageArray_gpu;
double* hittingNumArray_gpu;
unsigned_int* stageVertices_gpu;
unsigned int* locks;

unsigned_int vertexExp;
unsigned_int L;
unsigned_int dSize;

float* Fprev_gpu;
float* Fcurr_gpu;
// a host pointer pointing to the copy of D on the gpu
float* D_gpu;



__global__ cuda_select_vertex_wise(        
        byte* pick_gpu,
        byte* edgeArray_gpu,
        byte* stageArray_gpu,
        double* hittingNumArray_gpu,
        unsigned_int* stageVertices_gpu,
        unsigned int* locks,
        int size，
        double delta,
        double prob,
        int total
    ) {
    unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= size) {
        return;
    }

    unsigned_int i = stageVertices_gpu[thread_id];
    bool leaveLoop = false;

    while (!leaveLoop) {
        if (atomicExch(&locks[i], 1u) == 0u) {
            if ((pick_gpu[i] == false) && (hittingNumArray_gpu[i] > (pow(delta, 3) * total))) {
                pick_gpu[i] == true;
                stageArray_gpu[i] = 0;
                atomicAdd(&hittingCountStage, 1);
                atomicAdd(&pathCountStage, hittingNumArray_gpu[i]); 
                atomicExch(&locks[i], 0u);
                return;
            } else {
                leaveLoop = true;
                atomicExch(&locks[i], 0u);
            }
        }
    }
}


__global__ cuda_select_pair_wise(        
        byte* pick_gpu,
        byte* edgeArray_gpu,
        byte* stageArray_gpu,
        double* hittingNumArray_gpu,
        unsigned_int* stageVertices_gpu,
        unsigned int* locks,
        int size，
        double delta,
        double prob,
        int total
    ) {
    unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= size) {
        return;
    }

    unsigned int i = stageVertices_gpu[thread_id];
    bool leaveLoop = false;

    for (unsigned int jt = 0; jt < size; jt += 1) {
        j = stageVertices_gpu[jt];
        unsigned int small_lock = std::min(i, j);
        unsigned int large_lock = std::max(i, j);
        while (!leaveLoop) {
            if (i != j) {
                if ((atomicExch(&locks[small_lock], 1u) == 0u) && (atomicExch(&locks[large_lock], 1u) == 0u)) {
                    if (pick_gpu[i] == false) {
                        if (((double) rand() / (RAND_MAX)) <= prob_gpu) {
                            stageArray_gpu[i] = 0;
                            pick_gpu[i] = true;
                            atomicAdd(&hittingCountStage, 1);
                            atomicAdd(&pathCountStage, hittingNumArray_gpu[i]); 
                        }
                        if (pick[j] == false) {
                            if (((double) rand() / (RAND_MAX)) <= prob_gpu) {
                                stageArray_gpu[j] = 0;
                                pick_gpu[j] = true;
                                atomicAdd(&hittingCountStage, 1);
                                atomicAdd(&pathCountStage, hittingNumArray_gpu[j]); 
                                atomicExch(&locks[large_lock], 0u); 
                                atomicExch(&locks[small_lock], 0u); 
                                return;                                 
                            } else {
                                pick_gpu[i] = false;
                            }
                        }
                    } else {
                        atomicExch(&locks[large_lock], 0u); 
                        atomicExch(&locks[small_lock], 0u); 
                        return;                             
                    }
                    leaveLoop = true;
                    atomicExch(&locks[large_lock], 0u); 
                    atomicExch(&locks[small_lock], 0u);            
                }
            } else {
                if ((atomicExch(&locks[i], 1u) == 0u)) {
                    if (pick_gpu[i] == false) {
                        if (((double) rand() / (RAND_MAX)) <= prob) {
                            atomicAdd(&hittingCountStage, 1);
                            atomicAdd(&pathCountStage, hittingNumArray_gpu[i]); 
                        }
                    } else {
                        atomicExch(&locks[i], 0u);
                        return                          
                    }
                    leaveLoop = true;
                    atomicExch(&locks[i], 0u);                     
                }
            }
        }
        leaveLoop = false;
    }
}

__device__ void cuda_remove_edge(unsigned_int i) {
    /**
    Removes an edge from the graph.
    @param i: Index of edge.
    */
    if (edgeArray_gpu[i] == 1) {
        atomicSub(&edgeCount_gpu, 1);
    }
    edgeArray_gpu[i] = 0;
}


__global__ cuda_update_graph(        
        byte* pick_gpu,
        byte* edgeArray_gpu,
        byte* stageArray_gpu,
        double* hittingNumArray_gpu,
        unsigned_int* stageVertices_gpu,
        unsigned int* locks,
        int size，
        double delta,
        double prob,
        int total
    ) {
    unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= size) {
        return;
    }

    unsigned int i = stageVertices_gpu[thread_id];

    i = stageVertices[it];
    if (pick_gpu[i] == true) {
        cuda_remove_edge(i);
    }

}

__global__ void cuda_push_back_vector (

) {
    unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= edgeNum_gpu) {
        return;
    }
    if (edgeArray_gpu[thread_id] == 1) {
        while (true) {
            unsigned_int old_slot = array_slot;
            unsigned_int test_slot = atomicCAS(&array_slot, old_slot, old_slot-1);
            if (test_slot == old_slot) {
                stageVertices_gpu[test_slot-1] = (unsigned_int)thread_id;
                break;
            }
        }
    }
}

__global__ void cuda_update_size (

) {
    unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id >= edgeNum_gpu) {
        return;
    }
    if (edgeArray_gpu[thread_id] == 1) {
        atomicAdd(&size_gpu, 1);
    }    
}

void push_back_vector(

) {
    size_gpu = 0;
    int min_grid_size;
    int thread_block_size;

    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &thread_block_size,
        cuda_update_size,
        0,
        0
    );

    int grid_size = std::ceil(edgeNum_gpu / thread_block_size);

    cuda_update_size<<<grid_size, thread_block_size>>>();
    array_slot = size_gpu;
    cudaMalloc((void**)&stageVertices_gpu, size_gpu * sizeof(unsigned_int));

    cuda_push_back_vector<<<grid_size, thread_block_size>>>();
}

 
void pasha_gpu_init(
        byte* pick, 
        double alpha,
        double delta, 
        double epsilon, 
        double prob,
        double* hittingNumArray, 
        unsigned_int edgeNum,
        unsigned_int total,
        byte* edgeArray,
        byte* stageArray
) {
    
    prob_gpu = prob;
    edgeNum_gpu = edgeNum;
    edgeCount_gpu = edgeCount_gpu;
    alpha_gpu = alpha;
    delta_gpu = delta;
    epsilon_gpu = epsilon;
    total_gpu = tatal;

    cudaMalloc((void**)&pick_gpu, (unsigned_int)edgeNum * sizeof(byte));
    cudaMalloc((void**)&edgeArray_gpu, (unsigned_int)edgeNum * sizeof(byte));
    cudaMalloc((void**)&stageArray_gpu, (unsigned_int)edgeNum * sizeof(byte));
    cudaMalloc((void**)&hittingNumArray_gpu, (unsigned_int)edgeNum * sizeof(double));

    cudaMalloc((void**)&locks, (unsigned_int)edgeNum * sizeof(unsigned int));
    cudaMemset(locks,  0, (unsigned_int)edgeNum * sizeof(unsigned int));

    cudaMemcpy(pick_gpu, pick, (unsigned_int)edgeNum * sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(edgeArray_gpu, edgeArray, (unsigned_int)edgeNum * sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(stageArray_gpu, stageArray, (unsigned_int)edgeNum * sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(hittingNumArray_gpu, hittingNumArray, (unsigned_int)edgeNum * sizeof(double), cudaMemcpyHostToDevice);

}


void pasha_gpu_init_cont(
    unsigned_int LParam, 
    unsigned_int vertexExpParam, 
    unsigned_int dSizeParam
) {
    L = LParam;
    vertexExp = vertexExpParam;
    dSize = dSizeParam;

    cudaMalloc((void**)&D_gpu, dSize*sizeof(float));
    cudaMalloc((void**)&Fprev_gpu, vertexExp*sizeof(float));
    cudaMalloc((void**)&Fcurr_gpu, vertexExp*sizeof(float));

    cudaMemcpyToSymbol(d_vertexExp, &vertexExpParam, sizeof(unsigned_int));    
}


// assumes already inited
__device__ float D_get(float* D, int row, int col) {
    return D[row*d_vertexExp + col];
}
__device__ void D_set(float* D, int row, int col, float val) {
    D[row*d_vertexExp + col] = val;
}


__global__ void setInitialDFprev_gpu(float* D, float* Fprev) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= d_vertexExp) return;
    D_set(D, 0, tid, 1.4e-45);
    Fprev[tid] = 1.4e-45;
}

__global__ void calcNumStartingPathsOneIter_gpu(float* D, byte* edgeArray, int j) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= d_vertexExp) return;
    unsigned_int vertexExp2 = d_vertexExp * 2;
    unsigned_int vertexExp3 = d_vertexExp * 3;
    
    D_set(D, j, i, 
        edgeArray[i]*D_get(D, j-1, (i >> 2))
            + edgeArray[i + d_vertexExp]*D_get(D, j-1,((i + d_vertexExp) >> 2))
            + edgeArray[i + vertexExp2]*D_get(D, j-1,((i + vertexExp2) >> 2))
            + edgeArray[i + vertexExp3]*D_get(D, j-1,((i + vertexExp3) >> 2))
    );
}



void calcNumStartingPaths(byte* edgeArray, float* D, float* Fprev) {
    /**
    * This function generates D. D(v,i): # of i long paths starting from v after decycling
    */

    // cudaMemcpy(edgeArray_gpu, edgeArray, vertexExp*sizeof(byte), cudaMemcpyHostToDevice);

    int grid_size = 1 + ((vertexExp - 1) / NUM_THREADS);
    
    setInitialDFprev_gpu<<<grid_size, NUM_THREADS>>>(D_gpu, Fprev_gpu); 


    // TODO: replace loop with this https://towardsdatascience.com/gpu-optimized-dynamic-programming-8d5ba3d7064f
    for (unsigned_int j = 1; j <= L; j++) {
        calcNumStartingPathsOneIter_gpu<<<grid_size, NUM_THREADS>>>(D_gpu, edgeArray_gpu, j); 
        cudaDeviceSynchronize();
    }


    // cudaMemcpy(D, D_gpu, dSize*sizeof(float),  cudaMemcpyDeviceToHost);
    // cudaMemcpy(Fprev, Fprev_gpu, vertexExp*sizeof(float),  cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}




void pasha_gpu_compute(
        unsigned_int h
    ) {
    while (h > 0) {
        total_gpu = 0;
        hittingCountStage = 0;
        pathCountStage = 0;

        calculate_path_gpu(l, threads);

        imaxHittingNum = calculateHitting_number_parallel_gpu(l, true, threads);

        if (exit == -1) break;

        push_back_vector();

        int min_grid_size;
        int thread_block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &thread_block_size,
            cuda_select_vertex_wise,
            0,
            0
        );

        int grid_size = std::ceil(size_gpu / thread_block_size);

        cuda_select_vertex_wise<<<grid_size, thread_block_size>>>(
            pick_gpu,
            edgeArray_gpu,
            stageArray_gpu,
            hittingNumArray_gpu,
            stageVertices_gpu,
            locks,
            size,
            delta,
            prob,
            total
        );


        cuda_select_pair_wise<<<grid_size, thread_block_size>>>(
            pick_gpu,
            edgeArray_gpu,
            stageArray_gpu,
            hittingNumArray_gpu,
            stageVertices_gpu,
            locks,
            size,
            delta,
            prob,
            total
        );

        cudaDeviceSynchronize();

        hittingCount += hittingCountStage;

        if (pathCountStage >= hittingCountStage * pow((1.0 + epsilon), h) * (1 - 4*delta - 2*epsilon)) {
            cuda_update_graph<<<grid_size, thread_block_size>>>(
                pick_gpu,
                edgeArray_gpu,
                stageArray_gpu,
                hittingNumArray_gpu,
                stageVertices_gpu,
                locks,
                size,
                delta,
                prob,
                total
            );
            cudaDeviceSynchronize();
            h--;
        } else {
            hittingCount -= hittingCountStage;
        }

        cudaFree(stageVertices_gpu);
        // cudaMemcpy(pick, pick_gpu, (unsigned_int)edgeNum * sizeof(byte), cudaMemcpyDeviceToHost);
        // cudaMemcpy(edgeArray, edgeArray_gpu, (unsigned_int)edgeNum * sizeof(byte), cudaMemcpyDeviceToHost);
        // cudaMemcpy(stageArray, stageArray_gpu, (unsigned_int)edgeNum * sizeof(byte), cudaMemcpyDeviceToHost);
    }
}




void pasha_gpu_close(byte* pick) {
    cudaMemcpy(pick, pick_gpu, (unsigned_int)edgeNum_gpu * sizeof(byte), cudaMemcpyDeviceToHost);

    cudaFree(D_gpu);
    cudaFree(Fprev_gpu);
    cudaFree(Fcurr_gpu);

    cudaFree(pick_gpu);
    cudaFree(edgeArray_gpu);
    cudaFree(stageArray_gpu);
    cudaFree(hittingNumArray_gpu);
    cudaFree(stageVertices_gpu);
    cudaFree(locks);
}


