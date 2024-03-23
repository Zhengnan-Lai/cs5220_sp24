#include "common.h"
#include <mpi.h>
#include <algorithm> 
#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>

// Put any static global variables here that you will use throughout the simulation.

bool part_cmp(particle_t& x, particle_t& y) {return x.id < y.id;}
inline int min(int x, int y) {return x < y? x : y;}

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.vx += coef * dx * dt;
    particle.vy += coef * dy * dt;
    // neighbor.vx -= coef * dx * dt;
    // neighbor.vy -= coef * dy * dt;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    double pvx = p.vx;
    double pvy = p.vy;
    double px = p.x + pvx * dt;
    double py = p.y + pvy * dt;

    // Bounce from walls
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
}

// The length of a square bin
static double bin_length;
// The number of bins in one axis
static int num_bins;
// The number of rows per processor
static int num_rows_per_proc;
// The start of rows
static int row_st;
// The end of rows
static int row_ed;
// The indices of the particles
std::vector<int>* indices;
// The bins where particles are in
std::vector<particle_t> bins;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
    // std::cout<<"num_procs = "<<num_procs<<std::endl;
    bin_length = 1.1 * cutoff;
    num_bins = ceil(size / bin_length);
    num_rows_per_proc = (num_procs + num_bins - 1) / num_procs;  
    // std::cout<<"num_rows_per_proc = "<<num_rows_per_proc<<std::endl;
    // Each processor contains the rows in the interval [row_st, row_ed)
    row_st = rank * num_rows_per_proc;
    row_ed = min((rank+1) * num_rows_per_proc, num_bins);
    // std::cout<<"row_st = "<<row_st<<" , row_ed = "<<row_ed<<std::endl;
    indices = new std::vector<int>[(row_ed - row_st + 2) * num_bins];
    for(int i = 0; i < num_parts; i++){
        int x = parts[i].x / bin_length;
        if(x < row_st - 1 || x >= row_ed + 1) continue;
        int y = parts[i].y / bin_length;
        indices[(x-row_st+1) * num_bins + y].push_back(bins.size());
        bins.push_back(parts[i]);
    }
    // std::cout<<"bins_size = "<<bins.size()<<std::endl;
}

// TODO: two-way apply force, 2D implementation
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Apply forces
    for(particle_t& p: bins){
        int x = p.x / bin_length;
        if(x < row_st || x >= row_ed) continue;
        int y = p.y / bin_length;
        for(int i = x-1; i <= x+1; i++) for(int j = y-1; j <= y+1; j++){
            if(j < 0 || j >= num_bins) continue;
            for(int index: indices[(i-row_st+1) * num_bins + j])
                apply_force(p, bins[index]);
        }
    }
    for(int i = row_st - 1; i < row_ed + 1; i++) for(int j = 0; j < num_bins; j++){
        indices[(i-row_st+1) * num_bins + j].clear();
    }
    std::vector<particle_t> new_bins;
    std::vector<particle_t> top;
    std::vector<particle_t> bot;
    // Move particles
    for(particle_t& p: bins){
        int old_x = p.x / bin_length;
        if(old_x < row_st || old_x >= row_ed) continue;
        move(p, size);
        int x = p.x / bin_length;
        int y = p.y / bin_length;
        if(x >= row_st - 1 && x < row_ed + 1){
            indices[(x-row_st+1) * num_bins + y].push_back(new_bins.size());
            new_bins.push_back(p);
        }
        if(x <= row_st) top.push_back(p);
        else if(x >= row_ed - 1) bot.push_back(p);
    }
    // Communicate particles
    MPI_Status status;
    std::vector<particle_t> msg_from_top;
    std::vector<particle_t> msg_from_bot;
    if(rank % 2 == 0){ // Even columns send message
        if(rank > 0) MPI_Send(&top[0], top.size(), PARTICLE, rank-1, 0, MPI_COMM_WORLD);
        if(rank < num_procs - 1) MPI_Send(&bot[0], bot.size(), PARTICLE, rank+1, 1, MPI_COMM_WORLD);
    }
    else{
        int n1;
        MPI_Probe(rank-1, 1, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, PARTICLE, &n1);
        msg_from_top.resize(n1);
        MPI_Recv(&msg_from_top[0], n1, PARTICLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(rank < num_procs - 1){
            int n2; 
            MPI_Probe(rank+1, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &n2);
            msg_from_bot.resize(n2);
            MPI_Recv(&msg_from_bot[0], n2, PARTICLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    if(rank % 2 == 1){ // Odd columns send message
        MPI_Send(&top[0], top.size(), PARTICLE, rank-1, 2, MPI_COMM_WORLD);
        if(rank < num_procs - 1) MPI_Send(&bot[0], bot.size(), PARTICLE, rank+1, 3, MPI_COMM_WORLD);
    }
    else{
        if(rank > 0){
            int n1;
            MPI_Probe(rank-1, 3, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &n1);
            msg_from_top.resize(n1);
            MPI_Recv(&msg_from_top[0], n1, PARTICLE, rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if(rank < num_procs - 1){
            int n2; 
            MPI_Probe(rank+1, 2, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &n2);
            msg_from_bot.resize(n2);
            MPI_Recv(&msg_from_bot[0], n2, PARTICLE, rank+1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    for(particle_t& p: msg_from_top){
        int x = p.x / bin_length;
        int y = p.y / bin_length;
        if(x < row_st - 1 || x >= row_ed + 1) continue;
        indices[(x-row_st+1) * num_bins + y].push_back(new_bins.size());
        new_bins.push_back(p);
    }
    for(particle_t& p: msg_from_bot){
        int x = p.x / bin_length;
        int y = p.y / bin_length;
        if(x < row_st - 1 || x >= row_ed + 1) continue;
        indices[(x-row_st+1) * num_bins + y].push_back(new_bins.size());
        new_bins.push_back(p);
    }
    bins.clear();
    bins.swap(new_bins);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    std::vector<particle_t> send_parts;
    for(particle_t& p: bins){
        int x = p.x / bin_length;
        if(x >= row_st && x < row_ed) send_parts.push_back(p);
    }
    
    int n = send_parts.size();
    std::vector<int> count(num_procs);
    MPI_Gather(&n, 1, MPI_INT, &count[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> displ(num_procs);
    displ[0] = 0;
    for(int i = 1; i < num_procs; i++) displ[i] = displ[i-1] + count[i-1];
    MPI_Gatherv(&send_parts[0], send_parts.size(), PARTICLE, &parts[0], &count[0], &displ[0], PARTICLE, 0, MPI_COMM_WORLD);
    
    if(rank == 0) std::sort(parts, parts + num_parts, part_cmp);
}