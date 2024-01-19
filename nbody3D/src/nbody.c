//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // Include for memcpy
#include <omp.h>
#include <math.h>


//
typedef float              f32;
typedef double             f64;
typedef unsigned long long u64;

//
typedef struct particle_s {

  f32 x, y, z;
  f32 vx, vy, vz;
  
} particle_t;

//
void init(particle_t *p, u64 n)
{
  for (u64 i = 0; i < n; i++)
    {
      //
      u64 r1 = (u64)rand();
      u64 r2 = (u64)rand();
      f32 sign = (r1 > r2) ? 1 : -1;
      
      //
      p[i].x = sign * (f32)rand() / (f32)RAND_MAX;
      p[i].y = (f32)rand() / (f32)RAND_MAX;
      p[i].z = sign * (f32)rand() / (f32)RAND_MAX;

      //
      p[i].vx = (f32)rand() / (f32)RAND_MAX;
      p[i].vy = sign * (f32)rand() / (f32)RAND_MAX;
      p[i].vz = (f32)rand() / (f32)RAND_MAX;
    }
}

void move_particles(particle_t *p, const f32 dt, u64 n)
{
  //Used to avoid division by 0 when comparing a particle to itself
  const f32 softening = 1e-20;
  
  //For all particles
  for (u64 i = 0; i < n; i++)
    {
      //
      f32 fx = 0.0;
      f32 fy = 0.0;
      f32 fz = 0.0;

      //Newton's law: 17 FLOPs (Floating-Point Operations) per iteration
      for (u64 j = 0; j < n; j++)
	{ 
	  //3 FLOPs (Floating-Point Operations) 
	  const f32 dx = p[j].x - p[i].x; //1 (sub)
	  const f32 dy = p[j].y - p[i].y; //2 (sub)
	  const f32 dz = p[j].z - p[i].z; //3 (sub)

	  //Compute the distance between particle i and j: 6 FLOPs
	  const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening; //9 (mul, add)

	  //2 FLOPs (here, we consider pow to be 1 operation)
	  const f32 d_3_over_2 = pow(d_2, 3.0 / 2.0); //11 (pow, div)
	  
	  //Calculate net force: 6 FLOPs
	  fx += dx / d_3_over_2; //13 (add, div)
	  fy += dy / d_3_over_2; //15 (add, div)
	  fz += dz / d_3_over_2; //17 (add, div)
	}

      //Update particle velocities using the previously computed net force: 6 FLOPs 
      p[i].vx += dt * fx; //19 (mul, add)
      p[i].vy += dt * fy; //21 (mul, add)
      p[i].vz += dt * fz; //23 (mul, add)
    }

  //Update positions: 6 FLOPs
  for (u64 i = 0; i < n; i++)
    {
      p[i].x += dt * p[i].vx;
      p[i].y += dt * p[i].vy;
      p[i].z += dt * p[i].vz;
    }
}

//Version optimisée
void move_particles_opt(particle_t *p, const f32 dt, u64 n)
{
  const f32 softening = 1e-20;

  #pragma omp parallel for
  for (u64 i = 0; i < n; i++)
  {
    f32 fx = 0.0, fy = 0.0, fz = 0.0;

    // Manually unrolling the loop
    for (u64 j = 0; j < n; j += 2)  // Increment by 2 instead of 1
    {
      if (i != j) {
        const f32 dx = p[j].x - p[i].x;
        const f32 dy = p[j].y - p[i].y;
        const f32 dz = p[j].z - p[i].z;

        const f32 d_2 = dx * dx + dy * dy + dz * dz + softening;
        const f32 inv_d_3 = 1.0 / sqrt(d_2 * d_2 * d_2);

        fx += dx * inv_d_3;
        fy += dy * inv_d_3;
        fz += dz * inv_d_3;
      }

      // Check to avoid accessing out-of-bounds memory
      if (j + 1 < n && i != j + 1) {
        const f32 dx1 = p[j + 1].x - p[i].x;
        const f32 dy1 = p[j + 1].y - p[i].y;
        const f32 dz1 = p[j + 1].z - p[i].z;

        const f32 d_21 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + softening;
        const f32 inv_d_31 = 1.0 / sqrt(d_21 * d_21 * d_21);

        fx += dx1 * inv_d_31;
        fy += dy1 * inv_d_31;
        fz += dz1 * inv_d_31;
      }
    }

    p[i].vx += dt * fx;
    p[i].vy += dt * fy;
    p[i].vz += dt * fz;
  }

  for (u64 i = 0; i < n; i++)
  {
    p[i].x += dt * p[i].vx;
    p[i].y += dt * p[i].vy;
    p[i].z += dt * p[i].vz;
  }
}
  
f64 compute_delta_particle(particle_t *p_ref, particle_t *p, u64 n) { 
  f64 delta_pos = 0.0;
  f64 delta_vel = 0.0;

  for (u64 i = 0; i < n; i++) {
    delta_pos += sqrt(pow(p_ref[i].x - p[i].x, 2) + pow(p_ref[i].y - p[i].y, 2) + pow(p_ref[i].z - p[i].z, 2));
    delta_vel += sqrt(pow(p_ref[i].vx - p[i].vx, 2) + pow(p_ref[i].vy - p[i].vy, 2) + pow(p_ref[i].vz - p[i].vz, 2));
  }

  return (delta_pos + delta_vel) / (f64)(2 * n);
}

//

int main(int argc, char **argv)
{
    // Number of particles to simulate
    const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;

    // Number of experiments
    const u64 steps = 13;

    // Time step
    const f32 dt = 0.01;

    // Allocate memory for particles
    particle_t *p = malloc(sizeof(particle_t) * n);
    particle_t *p_opt = malloc(sizeof(particle_t) * n);

    // Initialize particles with the same initial conditions
    init(p, n);
    memcpy(p_opt, p, sizeof(particle_t) * n); // Copy initial state to p_opt

    // Performance metrics
    f64 rate_std = 0.0, rate_opt = 0.0;
    f64 drate_std = 0.0, drate_opt = 0.0;
    const u64 warmup = 3;
    const u64 s = sizeof(particle_t) * n;

    printf("\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);
    printf("\033[1m%5s %15s %15s %15s %15s %15s %15s\033[0m\n", "Step", "Time Std, s", "Time Opt, s", "Interact/s Std", "Interact/s Opt", "GFLOP/s Std", "GFLOP/s Opt"); 
    fflush(stdout);

    // Run and measure standard version
    for (u64 i = 0; i < steps; i++)
    {
        const f64 start_std = omp_get_wtime();
        move_particles(p, dt, n);
        const f64 end_std = omp_get_wtime();

        const f32 h1 = (f32)(n) * (f32)(n);
        const f32 h2 = (17.0 * h1 + 6.0 * (f32)n + 6.0 * (f32)n) * 1e-9;

        if (i >= warmup)
        {
            rate_std += h2 / (f32)(end_std - start_std);
            drate_std += (h2 * h2) / (f32)((end_std - start_std) * (end_std - start_std));
        }

        printf("%5llu %15.3e %15.3e %15.3e %15.3e %15.1f %15.1f\n",
               i,
               (end_std - start_std),
               0.0, // Placeholder for Time Opt, s
               h1 / (end_std - start_std),
               0.0, // Placeholder for Interact/s Opt
               h2 / (end_std - start_std),
               0.0); // Placeholder for GFLOP/s Opt
    }

    // Run and measure optimized version
    for (u64 i = 0; i < steps; i++)
    {
        const f64 start_opt = omp_get_wtime();
        move_particles_opt(p_opt, dt, n);
        const f64 end_opt = omp_get_wtime();

        const f32 h1 = (f32)(n) * (f32)(n);
        const f32 h2 = (17.0 * h1 + 6.0 * (f32)n + 6.0 * (f32)n) * 1e-9;

        if (i >= warmup)
        {
            rate_opt += h2 / (f32)(end_opt - start_opt);
            drate_opt += (h2 * h2) / (f32)((end_opt - start_opt) * (end_opt - start_opt));
        }

        printf("%5llu %15.3e %15.3e %15.3e %15.3e %15.1f %15.1f\n",
               i,
               0.0, // Placeholder for Time Std, s
               (end_opt - start_opt),
               0.0, // Placeholder for Interact/s Std
               h1 / (end_opt - start_opt),
               0.0, // Placeholder for GFLOP/s Std
               h2 / (end_opt - start_opt));
    }

    // Compute average performance metrics
    rate_std /= (f64)(steps - warmup);
    drate_std = sqrt(drate_std / (f64)(steps - warmup) - (rate_std * rate_std));

    rate_opt /= (f64)(steps - warmup);
    drate_opt = sqrt(drate_opt / (f64)(steps - warmup) - (rate_opt * rate_opt));

    printf("-----------------------------------------------------\n");
    printf("\033[1mStandard Version Performance: \033[0m%10.1lf +- %.1lf GFLOP/s\n", rate_std, drate_std);
    printf("\033[1mOptimized Version Performance: \033[0m%10.1lf +- %.1lf GFLOP/s\n", rate_opt, drate_opt);
    printf("-----------------------------------------------------\n");
    f64 delta = compute_delta_particle(p, p_opt, n);
    printf("Delta between standard and optimized versions: %f\n", delta);


    

    return 0;
}
