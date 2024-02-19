#include "common.h"
#include <cmath>
#include <cstdio>
#include <omp.h>

// Credit: stack exchange
typedef struct {
  particle_t **array;
  size_t used;
  size_t size;
} Bin;

void initBin(Bin *a, size_t initialSize) {
  a->array = (particle_t**) malloc(initialSize * sizeof(particle_t*));
  a->used = 0;
  a->size = initialSize;
}

void insertBin(Bin *a, particle_t& element) {
  if (a->used == a->size) {
    a->size *= 2;
    a->array = (particle_t**) realloc(a->array, a->size * sizeof(particle_t*));
  }
  a->array[a->used++] = &element;
}

void freeBin(Bin *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}

// The length of a square bin
static double bin_length;
// The number of bins (in one axis)
static int num_bins;
// The bins where particles are in
Bin* bins;
// Locks for each bin
omp_lock_t* locks;
// Compute the index at the (i,j) entry
inline int key(int i,int j) {return i + j * num_bins;}

#define min(a,b) a < b ? a : b
#define max(a,b) a > b ? a : b


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
    bin_length = 5 * cutoff;
    num_bins = ((int) (size / bin_length)) + 1;
    
    // bin_x = (int*) malloc(sizeof(int) * num_parts);
    // bin_y = (int*) malloc(sizeof(int) * num_parts);
    // The size of bins is num_bins x num_bins x num_parts
    bins = (Bin*) malloc(sizeof(Bin) * num_bins * num_bins);
    locks = (omp_lock_t*) malloc(sizeof(omp_lock_t) * num_bins * num_bins);
    for(int i = 0; i < num_bins; i++) for(int j = 0; j < num_bins; j++) {
        initBin(bins + key(i,j), 32);
        omp_init_lock(locks + key(i,j));
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset the number of particles in each bin
    #pragma omp for
    for(int i = 0; i < num_bins * num_bins; i++){
        bins[i].used = 0;
    }

    // Put the particles into bins
    #pragma omp for
    for(int i = 0; i < num_parts; i++){
        parts[i].ax = 0; parts[i].ay = 0;
        int index = key((int) (parts[i].x / bin_length), (int) (parts[i].y / bin_length));
        omp_set_lock(locks + index);
        insertBin(bins + index, parts[i]);
        omp_unset_lock(locks + index);
    }

    // Compute Forces
    #pragma omp for
    for(int i = 0; i < num_bins; i++){
        for(int j = 0; j < num_bins; j++){
            for(int dx = -1; dx <= 1; dx++){
                for(int dy = -1; dy <= 1; dy++){
                    if(i + dx < 0 || i + dx >= num_bins || j + dy < 0 || j + dy >= num_bins) continue;
                    int index1 = key(i,j), index2 = key(i+dx, j+dy);
                    for(int k = 0; k < bins[index1].used; k++){
                        for(int l = 0; l < bins[index2].used; l++){
                            apply_force(*(bins[index1].array[k]), *(bins[index2].array[l]));
                        }
                    }
                }
            }
        }
    }

    // Move Particles
    #pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
