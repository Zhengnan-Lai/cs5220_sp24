#include "common.h"
#include <cmath>
#include <cstdio>

// Credit: stack exchange
typedef struct {
  particle_t *array;
  size_t used;
  size_t size;
} Bin;

void initArray(Bin *a, size_t initialSize) {
  a->array = malloc(initialSize * sizeof(particle_t));
  a->used = 0;
  a->size = initialSize;
}

void insertArray(Bin *a, particle_t element) {
  // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
  // Therefore a->used can go up to a->size 
  if (a->used == a->size) {
    a->size *= 2;
    a->array = realloc(a->array, a->size * sizeof(particle_t));
  }
  a->array[a->used++] = element;
}

void freeArray(Bin *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}

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

#define min(a,b) a < b ? a : b


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
    // num_bins = ((int) min(cbrt(9 * num_parts), size / (1.1 * cutoff)));
    // bin_length = size / num_bins + 1e-8;

    bin_length = 1.1 * cutoff;
    num_bins = ((int) (size / bin_length)) + 1;
    
    bin_x = (int*) malloc(sizeof(int) * num_parts);
    bin_y = (int*) malloc(sizeof(int) * num_parts);
    // The size of bins is num_bins x num_bins x num_parts
    bins = (particle_t*) malloc(sizeof(particle_t) * num_bins * num_bins * num_parts);
    parts_in_bin = (int*) malloc(sizeof(int) * num_bins * num_bins);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset the number of particles in each bin
    for(int i = 0; i < num_bins; i++) for(int j = 0; j < num_bins; j++){
        parts_in_bin[key(i,j)] = 0;
    }
    // Put the particles into bins
    for(int i = 0; i < num_parts; i++){
        bin_x[i] = (int) (parts[i].x / bin_length);
        bin_y[i] = (int) (parts[i].y / bin_length);
        int index = key(bin_x[i], bin_y[i]);
        bins[index + parts_in_bin[index] * num_bins * num_bins] = parts[i];
        parts_in_bin[index]++;
    }
    // Compute Forces
    for (int i = 0; i < num_parts; ++i) {
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
    // for(int i = 0; i < num_parts; i++){
    //     parts[i].ax = 0; parts[i].ay = 0;
    //     for(int j = 0; j < num_parts; j++){
    //         apply_force(parts[i], parts[j]);
    //     }
    // }
    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
