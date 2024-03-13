#include "common.h"
#include <cuda.h>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define NUM_THREADS 320

// Put any static global variables here that you will use throughout the simulation.

int part_blks;
// The length of a square bin
double bin_length;
// The number of bins (in one axis)
int num_bins;
thrust::device_ptr<int> bin_ptr, prefix_bin_ptr;


// The length of a square bin on gpu
__device__ static double bin_length_gpu;
// The number of bins (in one axis) on gpus
__device__ static int num_bins_gpu;
__device__ static int num_parts_gpu;
int* part_id_gpu;
int* bin_id_gpu;
int* prefix_bin_id_gpu;

__inline__ __device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor, double px, double py, double nx, double ny) {
    double dx = nx - px;
    double dy = ny - py;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    atomicAdd(&neighbor.vx, -coef * dx * dt);
    atomicAdd(&neighbor.vy, -coef * dy * dt);
    atomicAdd(&particle.vx, coef * dx * dt);
    atomicAdd(&particle.vy, coef * dy * dt);
}

__global__ void move_gpu(particle_t* parts, double size, int* bin_id_gpu) {

    // Get thread (particle) ID
    int i = threadIdx.x + blockIdx.x * blockDim.x;  
    if(i >= num_parts_gpu) return;
    particle_t& p = parts[i];
    double px1 = p.x, py1 = p.y;
    int old_x = px1 / bin_length_gpu;
    int old_y = py1 / bin_length_gpu;
    int old_index = old_x + old_y * num_bins_gpu;
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    double pvx = p.vx;
    double pvy = p.vy;
    double px = px1 + pvx * dt;
    double py = py1 + pvy * dt;

    //
    //  bounce from walls
    //
    if(px < 0){
        int n_bounce = ceil(-px / size);
        if(n_bounce % 2){
            p.vx = -pvx;
            px = -px - (n_bounce - 1) * size;
        }
        else{
            px = n_bounce * size + px;
        }
    }
    else if(px > size){
        int n_bounce = ceil((px - size) / size);
        if(n_bounce % 2){
            p.vx = -pvx;
            px = (n_bounce + 1) * size - px;
        }
        else{
            px = px - n_bounce * size;
        }
    }
    p.x = px;

    if(py < 0){
        int n_bounce = ceil(-py / size);
        if(n_bounce % 2){
            p.vy = -pvy;
            py = -py - (n_bounce - 1) * size;
        }
        else{
            py = n_bounce * size + py;
        }
    }
    else if(py > size){
        int n_bounce = ceil((py - size) / size);
        if(n_bounce % 2){
            p.vy = -pvy;
            py = (n_bounce + 1) * size - py;
        }
        else{
            py = py - n_bounce * size;
        }
    }
    p.y = py;

    // Rebin
    int x = px / bin_length_gpu;
    int y = py / bin_length_gpu;
    int index = x + y * num_bins_gpu;
    if(old_index == index) return;
    atomicAdd(bin_id_gpu + index, 1);
    atomicSub(bin_id_gpu + old_index, 1);
    
}

__global__ void compute_forces_bin_gpu(particle_t* parts, int* part_id_gpu, int* bin_id_gpu) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(j >= num_parts_gpu) return;
    particle_t& particle = parts[part_id_gpu[j]];
    double px = particle.x, py = particle.y;
    int x = px / bin_length_gpu, y = py / bin_length_gpu;
    int i = x + y * num_bins_gpu;
    int ed1 = bin_id_gpu[i];
    int st2, ed2, st3, ed3, st4, ed4, st5, ed5;
    for(int k = j+1; k < ed1; k++){
        particle_t& neighbor = parts[part_id_gpu[k]];
        double nx = neighbor.x, ny = neighbor.y;
        apply_force_gpu(particle, neighbor, px, py, nx, ny);
    }
    // Compute forces for other 4 bins
    // dx = 1, dy = 0
    if(x + 1 < num_bins_gpu){
        st2 = ed1;
        ed2 = bin_id_gpu[i+1];
        for(int k = st2; k < ed2; k++){
            particle_t& neighbor = parts[part_id_gpu[k]];
            double nx = neighbor.x, ny = neighbor.y;
            apply_force_gpu(particle, neighbor, px, py, nx, ny);
        }
    }
    // dy = 1
    if(y + 1 >= num_bins_gpu) return;
    int u = x, v = y + 1;
    int index2 = u + v * num_bins_gpu;
    st4 = bin_id_gpu[index2-1];
    ed4 = bin_id_gpu[index2];
    // dx = -1
    if(x - 1 >= 0){
        st3 = bin_id_gpu[index2-2];
        ed3 = st4;
        for(int k = st3; k < ed3; k++){
            particle_t& neighbor = parts[part_id_gpu[k]];
            double nx = neighbor.x, ny = neighbor.y;
            apply_force_gpu(particle, neighbor, px, py, nx, ny);
        }
    }
    // dx = 0
    for(int k = st4; k < ed4; k++){
        particle_t& neighbor = parts[part_id_gpu[k]];
        double nx = neighbor.x, ny = neighbor.y;
        apply_force_gpu(particle, neighbor, px, py, nx, ny);
    }  
    // dx = 1
    if(x + 1 < num_bins_gpu){
        st5 = ed4;
        ed5 = bin_id_gpu[index2 + 1];
        for(int k = st5; k < ed5; k++){
            particle_t& neighbor = parts[part_id_gpu[k]];
            double nx = neighbor.x, ny = neighbor.y;
            apply_force_gpu(particle, neighbor, px, py, nx, ny);
        }
    }
    
}

__global__ void count_parts_gpu(particle_t* parts, int* bin_id_gpu) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num_parts_gpu) return;
    int x = parts[i].x / bin_length_gpu;
    int y = parts[i].y / bin_length_gpu;
    int index = x + y * num_bins_gpu;
    atomicAdd(bin_id_gpu + index, 1);
}

__global__ void insert_bins_gpu(particle_t* parts, int* part_id_gpu, int* prefix_bin_id_gpu) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= num_parts_gpu) return;
    int x = parts[i].x / bin_length_gpu;
    int y = parts[i].y / bin_length_gpu;
    int index = x + y * num_bins_gpu;
    int id = atomicAdd(prefix_bin_id_gpu + index, 1);
    part_id_gpu[id] = i;
}


void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    bin_length = 2.32 * cutoff;
    num_bins = ceil(size / bin_length);
    part_blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    cudaMemcpyToSymbol(bin_length_gpu, &bin_length, sizeof(double));
    cudaMemcpyToSymbol(num_bins_gpu, &num_bins, sizeof(int));
    cudaMemcpyToSymbol(num_parts_gpu, &num_parts, sizeof(int));
    cudaMalloc((void**)&part_id_gpu, sizeof(int) * num_parts);
    cudaMalloc((void**)&bin_id_gpu, sizeof(int) * (num_bins * num_bins));
    cudaMalloc((void**)&prefix_bin_id_gpu, sizeof(int) * (num_bins * num_bins + 1));
    cudaMemset(bin_id_gpu, 0, sizeof(int) * (num_bins * num_bins));
    cudaMemset(part_id_gpu, 0, sizeof(int) * num_parts);
    cudaMemset(prefix_bin_id_gpu, 0, sizeof(int) * (num_bins * num_bins + 1));
    

    // Count the number of particles in each bin
    count_parts_gpu<<<part_blks, NUM_THREADS>>>(parts, bin_id_gpu);
    
    // Prefix the bins
    bin_ptr = thrust::device_pointer_cast(bin_id_gpu);
    prefix_bin_ptr = thrust::device_pointer_cast(prefix_bin_id_gpu);
    thrust::exclusive_scan(bin_ptr, bin_ptr + num_bins * num_bins, prefix_bin_ptr);
    
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Asign particles into bins
    insert_bins_gpu<<<part_blks, NUM_THREADS>>>(parts, part_id_gpu, prefix_bin_id_gpu);

    // Compute forces
    compute_forces_bin_gpu<<<part_blks, NUM_THREADS>>>(parts, part_id_gpu, prefix_bin_id_gpu);

    // Move particles
    move_gpu<<<part_blks, NUM_THREADS>>>(parts, size, bin_id_gpu);

    // Recompute prefix
    bin_ptr = thrust::device_pointer_cast(bin_id_gpu);
    prefix_bin_ptr = thrust::device_pointer_cast(prefix_bin_id_gpu);
    thrust::exclusive_scan(bin_ptr, bin_ptr + num_bins * num_bins, prefix_bin_ptr);
}