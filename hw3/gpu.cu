#include "common.h"
#include <cuda.h>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.

int part_blks, bin_blks;
int devId, num_procs;
// The length of a square bin
static double bin_length;
// The number of bins (in one axis)
static int num_bins;


// The length of a square bin on gpu
__device__ static double bin_length_gpu;
// The number of bins (in one axis) on gpus
__device__ static int num_bins_gpu;
int* part_id_gpu;
int* bin_id_gpu;
int* prefix_bin_id_gpu;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    // apply force to both particles at once
    atomicAdd(&particle.ax, coef * dx);
    atomicAdd(&particle.ay, coef * dy);
    atomicAdd(&neighbor.ax, -coef * dx);
    atomicAdd(&neighbor.ay, -coef * dy);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size, int* bin_id_gpu) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    for(int i = tid; i < num_parts; i += offset){
        particle_t* p = &particles[i];
        int old_x = p->x / bin_length_gpu;
        int old_y = p->y / bin_length_gpu;
        int old = old_x + old_y * num_bins_gpu;
        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;

        //
        //  bounce from walls
        //
        while (p->x < 0 || p->x > size) {
            p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
            p->vx = -(p->vx);
        }
        while (p->y < 0 || p->y > size) {
            p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
            p->vy = -(p->vy);
        }

        // Rebin
        int x = p->x / bin_length_gpu;
        int y = p->y / bin_length_gpu;
        int index = x + y * num_bins_gpu;
        if(old != index){
            atomicSub(bin_id_gpu + old, 1);
            atomicAdd(bin_id_gpu + index, 1);
        }

        // reset accelerations here
        p->ax = p->ay = 0;
    }   
}

__global__ void compute_forces_bin_gpu(particle_t* parts, int num_parts, int* part_id_gpu, int* bin_id_gpu) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    for(int tid = t; tid < num_bins_gpu * num_bins_gpu; tid += offset){
        // Compute forces inside the same bin
        int st = tid == 0? 0 : bin_id_gpu[tid-1];
        int ed = bin_id_gpu[tid];
        for(int j = st; j < ed; j++) for(int k = j+1; k < ed; k++){
            apply_force_gpu(parts[part_id_gpu[j]], parts[part_id_gpu[k]]);
        }
        // Compute forces for other 4 bins; loop unrolling to fasten 
        int x = tid % num_bins_gpu, y = tid / num_bins_gpu;
        // dx = 1, dy = -1
        if(x + 1 < num_bins_gpu && y - 1 >= 0){
            int u = x + 1, v = y - 1;
            int index2 = u + v * num_bins_gpu;
            int st2 = index2 == 0? 0 : bin_id_gpu[index2-1];
            int ed2 = bin_id_gpu[index2];
            for(int j = st; j < ed; j++) for(int k = st2; k < ed2; k++){
                apply_force_gpu(parts[part_id_gpu[j]], parts[part_id_gpu[k]]);
            }
        }
        // dx = 1, dy = 0
        if(x + 1 < num_bins_gpu){
            int u = x + 1, v = y;
            int index2 = u + v * num_bins_gpu;
            int st2 = index2 == 0? 0 : bin_id_gpu[index2-1];
            int ed2 = bin_id_gpu[index2];
            for(int j = st; j < ed; j++) for(int k = st2; k < ed2; k++){
                apply_force_gpu(parts[part_id_gpu[j]], parts[part_id_gpu[k]]);
            }
        }
        // dx = 1, dy = 1
        if(x + 1 < num_bins_gpu && y + 1 < num_bins_gpu){
            int u = x + 1, v = y + 1;
            int index2 = u + v * num_bins_gpu;
            int st2 = index2 == 0? 0 : bin_id_gpu[index2-1];
            int ed2 = bin_id_gpu[index2];
            for(int j = st; j < ed; j++) for(int k = st2; k < ed2; k++){
                apply_force_gpu(parts[part_id_gpu[j]], parts[part_id_gpu[k]]);
            }
        }
        // dx = 0, dy = 1
        if(y + 1 < num_bins_gpu){
            int u = x, v = y + 1;
            int index2 = u + v * num_bins_gpu;
            int st2 = index2 == 0? 0 : bin_id_gpu[index2-1];
            int ed2 = bin_id_gpu[index2];
            for(int j = st; j < ed; j++) for(int k = st2; k < ed2; k++){
                apply_force_gpu(parts[part_id_gpu[j]], parts[part_id_gpu[k]]);
            }
        }
    }
}

__global__ void count_parts_gpu(particle_t* parts, int num_parts, int* part_id_gpu, int* bin_id_gpu) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;  
    for(int i = tid; i < num_parts; i += offset){
        // reset accelerations here
        parts[i].ax = parts[i].ay = 0;
        int x = parts[i].x / bin_length_gpu;
        int y = parts[i].y / bin_length_gpu;
        int index = x + y * num_bins_gpu;
        atomicAdd(bin_id_gpu + index, 1);
    }
}

__global__ void insert_bins_gpu(particle_t* parts, int num_parts, int* part_id_gpu, int* bin_id_gpu, int* prefix_bin_id_gpu) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    for(int i = tid; i < num_parts; i += offset){
        int x = parts[i].x / bin_length_gpu;
        int y = parts[i].y / bin_length_gpu;
        int index = x + y * num_bins_gpu;
        int id = atomicAdd(prefix_bin_id_gpu + index, 1);
        part_id_gpu[id] = i;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    // For n = 1e6, c = 1.5 ~ 2
    // bin_length = 2 * cutoff;
    bin_length = size / sqrt(num_parts);
    num_bins = ceil(size / bin_length);

    cudaMemcpyToSymbol(bin_length_gpu, &bin_length, sizeof(double));
    cudaMemcpyToSymbol(num_bins_gpu, &num_bins, sizeof(int));
    cudaMalloc((void**)&part_id_gpu, sizeof(int) * num_parts);
    cudaMalloc((void**)&bin_id_gpu, sizeof(int) * (num_bins * num_bins));
    cudaMalloc((void**)&prefix_bin_id_gpu, sizeof(int) * (num_bins * num_bins + 1));
    cudaMemset(part_id_gpu, 0, sizeof(int) * num_parts);
    cudaMemset(bin_id_gpu, 0, sizeof(int) * (num_bins * num_bins));
    cudaMemset(bin_id_gpu, 0, sizeof(int) * (num_bins * num_bins + 1));

    cudaGetDevice(&devId);
    cudaDeviceGetAttribute(&num_procs, cudaDevAttrMultiProcessorCount, devId);
    part_blks = min(512, (num_parts + NUM_THREADS - 1) / NUM_THREADS);
    bin_blks =  (num_bins * num_bins + NUM_THREADS - 1) / NUM_THREADS;

    // Count the number of particles in each bin
    count_parts_gpu<<<part_blks, NUM_THREADS>>>(parts, num_parts, part_id_gpu, bin_id_gpu);
    // Prefix the bins
    thrust::device_ptr<int> bin_ptr = thrust::device_pointer_cast(bin_id_gpu);
    thrust::device_ptr<int> prefix_bin_ptr = thrust::device_pointer_cast(prefix_bin_id_gpu);
    thrust::exclusive_scan(bin_ptr, bin_ptr + num_bins * num_bins, prefix_bin_ptr);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Asign particles into bins
    insert_bins_gpu<<<part_blks, NUM_THREADS>>>(parts, num_parts, part_id_gpu, bin_id_gpu, prefix_bin_id_gpu);

    // Compute forces
    compute_forces_bin_gpu<<<bin_blks, NUM_THREADS>>>(parts, num_parts, part_id_gpu, prefix_bin_id_gpu);

    // Move particles
    move_gpu<<<part_blks, NUM_THREADS>>>(parts, num_parts, size, bin_id_gpu);

    // Recompute prefix
    thrust::device_ptr<int> bin_ptr = thrust::device_pointer_cast(bin_id_gpu);
    thrust::device_ptr<int> prefix_bin_ptr = thrust::device_pointer_cast(prefix_bin_id_gpu);
    thrust::exclusive_scan(bin_ptr, bin_ptr + num_bins * num_bins, prefix_bin_ptr);
}
