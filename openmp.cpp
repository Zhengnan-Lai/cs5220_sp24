#include "common.h"
#include <cstdio>
#include <cmath>
#include <omp.h>

// The length of a square bin
static double bin_length;
// The number of bins (in one axis)
static int num_bins;
// The bins where particles are in
particle_t* bins;
// The indices of bins each particle is in
int* bin_x; int* bin_y;
// The number of particles in each bin
int* parts_in_bin;
// Compute the index at the (i,j) entry
inline int key(int i,int j) {return i + j * num_bins;}



// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here

    // Legal range is [0, size / bin_length]
    bin_length = 1.1 * cutoff;
    num_bins = ((int) (size / bin_length)) + 1;
    bin_x = (int*) malloc(sizeof(int) * num_parts);
    bin_y = (int*) malloc(sizeof(int) * num_parts);
    // The size of bins is num_bins x num_bins x num_parts
    bins = (particle_t*) malloc(sizeof(particle_t) * num_bins * num_bins * num_parts);
    parts_in_bin = (int*) malloc(sizeof(int) * num_bins * num_bins);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
	int id = omp_get_thread_num();
	int nthrds = omp_get_num_threads();

	// Parallelly reset the number of particles in each bin
	for(int i = id; i < num_bins; i += nthrds) for(int j = 0; j < num_bins; j++){
		parts_in_bin[key(i,j)] = 0;
	}

	#pragma omp barrier
	// Parallelly put the particles into bins
	for(int i = id; i < num_parts; i += nthrds){
		bin_x[i] = (int) (parts[i].x / bin_length);
		bin_y[i] = (int) (parts[i].y / bin_length);
		int index = key(bin_x[i], bin_y[i]);
		// printf("binx = %d, biny = %d, num_bins = %d\n", bin_x[i], bin_y[i], num_bins);
		// printf("index = %d, parts = %d\n", index, parts_in_bin[index]);
		// printf("size = %d\n", num_bins * num_bins * num_parts);
		bins[index + parts_in_bin[index] * num_bins * num_bins] = parts[i];
		parts_in_bin[index]++;
	}
	
	#pragma omp barrier
    // Parallelly compute forces
	for (int i = id; i < num_parts; i += nthrds) {
		parts[i].ax = parts[i].ay = 0;
		for(int j = -1; j <= 1; j++) for(int k = -1; k <= 1; k++){
			// Discard illegal bins
			if(bin_x[i] + j < 0 || bin_x[i] + j >= num_bins || bin_y[i] + k < 0 || bin_y[i] + k >= num_bins) continue;
			int index = key(bin_x[i]+j, bin_y[i]+k);
			for(int l = 0; l < parts_in_bin[index]; l++){
				apply_force(parts[i], bins[index + l * num_bins * num_bins]);
			}
		}
	}

	#pragma omp barrier
	// Parallelly move particles
	for (int i = id; i < num_parts; i += nthrds) {
		move(parts[i], size);
	}
}
