#ifndef __HITTING_NUM_CUH__
#define __HITTING_NUM_CUH__

using unsigned_int = uint64_t;
using byte = uint8_t;


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
    );

void pasha_gpu_compute(
        unsigned_int h
    );

void pasha_gpu_close(
        byte* pick
    );