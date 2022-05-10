using namespace std;
using unsigned_int = uint64_t;
using byte = uint8_t;

__global__ cuda_func_1(        
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


__global__ cuda_func_2(        
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
                        if (((double) rand() / (RAND_MAX)) <= prob) {
                            stageArray_gpu[i] = 0;
                            pick_gpu[i] = true;
                            atomicAdd(&hittingCountStage, 1);
                            atomicAdd(&pathCountStage, hittingNumArray_gpu[i]); 
                        }
                        if (pick[j] == false) {
                            if (((double) rand() / (RAND_MAX)) <= prob) {
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

__device__ void removeEdge(unsigned_int i) {
    /**
    Removes an edge from the graph.
    @param i: Index of edge.
    */
    if (edgeArray[i] == 1) {
        atomicSub(&edgeCount, 1);
    }
    edgeArray[i] = 0;
}

__global__ cuda_func_3(        
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
        removeEdge(i);
        /* directly updata memory */
        string label = getLabel(i);
        hittingStream << label << "\n";
        /* directly updata memory */
    }

}


    unsigned_int HittingRandomParallel(unsigned_int L, const char *hittingPath, unsigned_int threads) {

        calculatePaths(l, threads);
        unsigned_int imaxHittingNum = calculateHittingNumberParallel(l, false, threads);
        cout << "Max hitting number: " << hittingNumArray[imaxHittingNum] << endl;
        h = findLog((1.0+epsilon), hittingNumArray[imaxHittingNum]);
        double prob = delta/(double)l;

        while (h > 0) {

            //cout << h << endl;
            total = 0;
            unsigned_int hittingCountStage = 0;
            double pathCountStage = 0;
            // calculatePaths(l, threads);
            cuda_calculatePaths(l, threads);
            // imaxHittingNum = calculateHittingNumberParallel(l, true, threads);
            imaxHittingNum = cuda_calculateHittingNumberParallel(l, true, threads);
            if (exit == -1) break;
            // stageVertices = pushBackVector();
            stageVertices = cuda_pushBackVector();

            std::size_t size = stageVertices.size();

            int min_grid_size;
            int thread_block_size;
            cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size,
                &thread_block_size,
                set_uhs_frequencies_from_sample_gpu,
                0,
                0
            );
            int grid_size = std::ceil(size / thread_block_size);

            cuda_func_1<<<grid_size, thread_block_size>>>(
                stageVertices, 
                size, 
                locks
            );


            cuda_func_2<<<grid_size, thread_block_size>>>(
                stageVertices,
                size, 
                locks
            );

            cudaDeviceSynchronize();

            hittingCount += hittingCountStage;

            if (pathCountStage >= hittingCountStage * pow((1.0 + epsilon), h) * (1 - 4*delta - 2*epsilon)) {
                cuda_func_3<<<grid_size, thread_block_size>>>(
                    stageVertices, 
                    size, 
                    locks
                );
                cudaDeviceSynchronize();
                h--;
            } else {
                hittingCount -= hittingCountStage;
            }

        }
        hittingStream.close();
        // topologicalSort();
        // cout << "Length of longest remaining path: " <<  maxLength() << "\n";
        return hittingCount;
    }



    int calculatePaths(unsigned_int L, unsigned_int threads) {
    /**
    Calculates number of L-k+1 long paths for all vertices.
    @param L: Sequence length.
    @return 1: True if path calculation completes.
    */
        omp_set_dynamic(0);
        curr = 1;
        vertexExp2 = vertexExp * 2;
        vertexExp3 = vertexExp * 3;
        vertexExpMask = vertexExp - 1;
        vertexExp_1 = pow(ALPHABET_SIZE, k-2);

        #pragma omp parallel for num_threads(threads)
        for (unsigned_int i = 0; i < vertexExp; i++) {D[0][i] = 1.4e-45; Fprev[i] = 1.4e-45;}
        for (unsigned_int j = 1; j <= L; j++) {
            #pragma omp parallel for num_threads(threads)
            for (unsigned_int i = 0; i < vertexExp; i++) {
                D[j][i] = edgeArray[i]*D[j-1][(i >> 2)] + edgeArray[i + vertexExp]*D[j-1][((i + vertexExp) >> 2)] + edgeArray[i + vertexExp2]*D[j-1][((i + vertexExp2) >> 2)] + edgeArray[i + vertexExp3]*D[j-1][((i + vertexExp3) >> 2)];
                //cout << (float)(Dval[j][i] * pow(2, Dexp[j][i])) << endl;
                //D[j][i] = Dval[i];
            }
        }
        
        #pragma omp parallel for num_threads(threads)
        for (unsigned_int i = 0; i < (unsigned_int)edgeNum; i++) hittingNumArray[i] = 0;
        while (curr <= L) {
            #pragma omp parallel for num_threads(threads)
            for (unsigned_int i = 0; i < vertexExp; i++) {
                unsigned_int index = (i * 4);
                Fcurr[i] = (edgeArray[index]*Fprev[index & vertexExpMask] + edgeArray[index + 1]*Fprev[(index + 1) & vertexExpMask] + edgeArray[index + 2]*Fprev[(index + 2) & vertexExpMask] + edgeArray[index + 3]*Fprev[(index + 3) & vertexExpMask]);
            
            }
            #pragma omp parallel for num_threads(threads)
            for (unsigned_int i = 0; i < (unsigned_int)edgeNum; i++) {
                hittingNumArray[i] += (Fprev[i % vertexExp]/1.4e-45) * (D[(L-curr)][i / ALPHABET_SIZE]/1.4e-45);
                if (edgeArray[i] == 0) hittingNumArray[i] = 0;
            }
            #pragma omp parallel for num_threads(threads)
            for (unsigned_int i = 0; i < vertexExp; i++) Fprev[i] = Fcurr[i];
            curr++;
        }
        return 1;
    }

    int findLog(double base, double x) {
    /**
    Finds the logarithm of a given number with respect to a given base.
    @param base: Base of logartihm, x: Input number.
    @return (int)(log(x) / log(base)): Integer logarithm of the number and the given base.
    */
        return (int)(log(x) / log(base));
    }
    vector<unsigned_int> pushBackVector() {
        vector<unsigned_int> stageVertices;
        for(unsigned_int i = 0; i < (unsigned_int)edgeNum; i++) {
            if (stageArray[i] == 1) stageVertices.push_back(i);
        }
        return stageVertices;
    }