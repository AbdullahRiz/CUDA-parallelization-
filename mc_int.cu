#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <util_bins.h>
#include <util_cuda.h>
#include <util_random.h>
#include <util_test.h>

/* set the dimension of the domain over which the integration is performed
 * TODO use all of these dimensions to test your code:
 * - 1
 * - 2
 * - 4
 */
#define DIM 4

/* set the min and max values of the integration domain (for all dimensions) */
#define DOMAIN_MIN 0.0
#define DOMAIN_MAX 1.0

/* set the number bins per dimension
 * TODO set according to:
 * - use `8` for development and debugging
 * - use `64` for "real" calculations after debugging
 */
#define N_BINS_PER_DIM 64

/* set the number of CUDA threads per block */
#define N_THREADS_PER_BLOCK 256

/**
 * Evaluates the integrand for the domain [DOMAIN_MIN, DOMAIN_MAX]^DIM.
 * Uses a different integrand depending on the dimension.
 */
__host__ __device__ double integrand_fn(double *x) {
#if (DIM == 1)
  const double c00 = 0.33;
  const double c10 = 0.67;
  const double d0 = 100.0;
  const double d1 = 100.0;
  return exp(-d0*(x[0] - c00)*(x[0] - c00)) + 
         exp(-d1*(x[0] - c10)*(x[0] - c10));
#elif (DIM == 2)
  const double c00 = 0.33;
  const double c10 = 0.67;
  const double c01 = 0.5;
  const double c11 = 0.5;
  const double d0 = 100.0;
  const double d1 = 100.0;
  return exp(-d0*( (x[0] - c00)*(x[0] - c00) + (x[1] - c01)*(x[1] - c01) )) + 
         exp(-d1*( (x[0] - c10)*(x[0] - c10) + (x[1] - c11)*(x[1] - c11) ));
#elif (DIM == 4)
  const double c00 = 0.33;
  const double c10 = 0.67;
  const double c01 = 0.5;
  const double c11 = 0.5;
  const double c02 = 0.5;
  const double c12 = 0.5;
  const double c03 = 0.5;
  const double c13 = 0.5;
  const double d0 = 100.0;
  const double d1 = 100.0;
  return exp(-d0*( (x[0] - c00)*(x[0] - c00) + (x[1] - c01)*(x[1] - c01) + (x[2] - c02)*(x[2] - c02) + (x[3] - c03)*(x[3] - c03) )) + 
         exp(-d1*( (x[0] - c10)*(x[0] - c10) + (x[1] - c11)*(x[1] - c11) + (x[2] - c12)*(x[2] - c12) + (x[3] - c13)*(x[3] - c13) ));
#else
#error "Dimension is not supported."
#endif
}

void integral_reference(double *int_approx, double *error_estimate) {
#if (DIM == 1)
  *int_approx     = 0.35449022821615284;
  *error_estimate = 1.625574217470555e-13;
#elif (DIM == 2)
  *int_approx     = 0.06283175701091294;
  *error_estimate = 1.4509634819439616e-13;
#elif (DIM == 4)
  *int_approx     = 0.001973917862370161;
  *error_estimate = 1.4899489211279838e-13;
#endif
}

/**
 * Calculates the bin index from a bin's multi-dimensional coordinates.
 *
 * Input:
 *   coord    Multi-dimensonal coordinates, integer-valued (array size is `DIM`)
 *
 * Return:
 *   Integer valued index corresponding to one bin (range: 0 <= bin_id < N_BINS_PER_DIM^DIM)
 */
__host__ __device__ int get_bin_index_from_coordinates(int *coord) {
  const int dim = DIM;
  const int n_bins_per_dim = N_BINS_PER_DIM;

  // TODO
 int i = 0;
 
 double bin_id = 0;
 for(i = 0;i < DIM; i++){
   bin_id += coord[i]*pow(n_bins_per_dim, i);
 }
 if(0<= bin_id && bin_id < pow(n_bins_per_dim, dim))
    {  return bin_id;}
 else{
   return -1;}

 
  
}

/**
 * Calculates multi-dimensional coordinates of the input bin index.
 * This reverses the function `get_bin_index_from_coordinates`.
 *
 * Input:
 *   bin_id   Integer valued corresponding to one bin (range: 0 <= bin_id < N_BINS_PER_DIM^DIM)
 *
 * Output:
 *   coord    Multi-dimensonal coordinates, integer-valued (array size is `DIM`)
 */
__host__ __device__ void get_bin_coordinates_from_index(int *coord, int bin_id) {
  const int dim = DIM;
  const int n_bins_per_dim = N_BINS_PER_DIM;
  
  int bi; 
  memcpy(&bi, &bin_id, sizeof(int));

  // TODO
 int i = 0;
 coord[dim-1] = bin_id/pow(n_bins_per_dim, dim - 1);
 for(i = dim-2; i >= 0; i--){
  bi = fmod((double)bi, pow( n_bins_per_dim, i +1));
  
  


  coord[i] = (int) ((double) bi/pow(n_bins_per_dim, i));

 }


}

/**
 * Calculates the index offset to access bin ranges corresponding to the
 * dimension `dim_id`.
 *
 * Input:
 *   dim_id           Index corresponding to dimension (range: 0 <= dim_id < DIM)
 *
 * Return:
 *   Index offset for bin ranges array.
 */
__host__ __device__ int get_offset_for_bin_range_1dim(int dim_id) {
  const int n_bins_per_dim = N_BINS_PER_DIM;

  return dim_id*(n_bins_per_dim + 1);
}

/**
 * Extracts the min and max values of a bin's bounding box from bin ranges.
 * The bin is specified with multi-dimenional coordinates.
 *
 * Input:
 *   bin_ranges       Boundaries between bins (array size: `DIM*(N_BINS_PER_DIM+1)`)
 *   coord            Multi-dimensonal coordinates, integer-valued (array size is `DIM`)
 *
 * Output:
 *   bin_min          Min values of the bin's bounding box
 *   bin_max          Max values of the bin's bounding box
 */
__host__ __device__ void get_bin_min_and_max(double *bin_min, double *bin_max,
                                             const double *bin_ranges, int *coord) {
  const int dim = DIM;
  int dim_id, offset;

  for (dim_id=0; dim_id<dim; dim_id++) {
    offset = get_offset_for_bin_range_1dim(dim_id);
    bin_min[dim_id] = bin_ranges[offset + coord[dim_id]];
    bin_max[dim_id] = bin_ranges[offset + coord[dim_id] + 1];
  }
}

/**
 * Calculates the volume of a bin from its bounding box.
 *
 * Input:
 *   bin_min  Min values of the bin's bounding box (array size is `DIM`)
 *   bin_max  Max values of the bin's bounding box (array size is `DIM`)
 *
 * Return:
 *   Volume of one bin (as double type).
 */
__host__ __device__ double get_bin_volume(const double *bin_min, const double *bin_max) {
  const int dim = DIM;

  // TODO
  int i = 0;
  double v = 1.0;
  for(i = 0; i < dim;i++){
    
  v *= bin_max[i]-bin_min[i];
  }
  return v;

}

/******************** CPU version ********************/

/**
 * Generates a multi-dimensional random number inside a bin.
 *
 * Input:
 *   bin_min  Min values of the bin's bounding box (array size is `DIM`)
 *   bin_max  Max values of the bin's bounding box (array size is `DIM`)
 *
 * Output:
 *   x        Random values inside bin bounds (array size is `DIM`)
 */
void random_fn(double *x, const double *bin_min, const double *bin_max) {
  const int dim = DIM;
  int dim_id;

  for (dim_id=0; dim_id<dim; dim_id++) {
    x[dim_id] = (bin_max[dim_id] - bin_min[dim_id]) * random_uniform() + bin_min[dim_id];
  }
}

/**
 * Performs Monte Carlo integration in multi-dimensions of the function
 * `integrand_fn`.  The approximation of the integrand is computed along with
 * an approximated variance, which is interpreted as the (square of the)
 * integration error.
 *
 * Input:
 *   n_mc_evals   Number of Monte Carlo evaluations (i.e., samples)
 *   n_bin_evals  Number of Monte Carlo evaluations per bin
 *   bin_ranges   Boundaries between bins (array size: `DIM*(N_BINS_PER_DIM+1)`)
 *
 * Output:
 *   int_approx  Pointer(!) to Approximate value of the integration
 *   var_approx  Pointer(!) to Approximate value of the variance (i.e., error^2 of integration)
 *   bin_content Values of integration for each bin (array size: `N_BINS_PER_DIM^DIM`)
 */
void mc_int_epoch_sequential(double *int_approx, double *var_approx, double *bin_content,
                             unsigned long n_mc_evals, unsigned long n_bin_evals, const double *bin_ranges) {
  
  const int dim = DIM;
  const int n_bins_per_dim = N_BINS_PER_DIM;
  const int n_bins_total   = pow(n_bins_per_dim, dim);

  int coord[DIM];
  double bin_min[DIM], bin_max[DIM];

  double x[DIM];

  double Sum = 0;
  double Sumvar = 0;

  // TODO (most code is missing but comments are placed for orientation)
    

  /* run Monte Carlo over bins */
  for (int bin_id=0; bin_id<n_bins_total; bin_id++) {
    
    /* get bin parameters */
    get_bin_coordinates_from_index(coord, bin_id);
    get_bin_min_and_max(bin_min, bin_max, bin_ranges, coord);
   
    


     double bin_volume = get_bin_volume(bin_min, bin_max);
   
    

    double f = 0;
    double ff=0;
    /* evaluate/sample integrand */
      for (int j=0; j<n_bin_evals; j++) {
        random_fn(x, bin_min, bin_max);
      
      
      // last part of approximate formula
        ff += integrand_fn(x) * integrand_fn(x);
    //SUM_{j IN [0,n_bin_evals)} f(x) 
        f += integrand_fn(x);
    }

    /* store intermediate result */
    
    // TODO set output bin_content[bin_id]
    
    
    // bin content formula
    bin_content[bin_id] = (n_bins_total * bin_volume * f);
    // second part of formula
    Sum += n_bins_total*bin_volume * f;
    Sumvar += n_bins_total*bin_volume *ff;
    /* add bin contributions to averages */
      
  }

  /* calculate approximate integral and variance */
  // TODO set output *int_approx
  *int_approx = (1.0/ n_mc_evals) * Sum;
  
  // TODO set output *var_approx
  *var_approx = Sumvar/(n_mc_evals - 1) - pow(*int_approx, 2)/(n_mc_evals - 1);
}

/**
 * Sums up contributions from each bin and scales with `1/n_mc_evals`.
 *
 * Input:
 *   bin_content Value of integration for each bin (array size: `N_BINS_PER_DIM^DIM`)
 *   n_mc_evals   Number of Monte Carlo evaluations (i.e., samples)
 *
 * Output:
 *   int_approx  Approximate value of the integration
 */
void reduce_bin_content_to_integral(double *int_approx,
                                    double *bin_content, unsigned long n_mc_evals) {
  const int dim = DIM;
  const int n_bins_per_dim = N_BINS_PER_DIM;
  const int n_bins_total   = pow(n_bins_per_dim, dim);
  int bin_id;
  double avg_scale = 1.0 / ((double) n_mc_evals);

  for (bin_id=0; bin_id<n_bins_total; bin_id++) {
    *int_approx += avg_scale * bin_content[bin_id];
  }
}

/******************** CPU version ********************/

/******************** GPU version ********************/

/**
 * Generates a multi-dimensional random number inside a bin.
 *
 * Input:
 *   bin_min      Min values of the bin's bounding box (array size is `DIM`)
 *   bin_max      Max values of the bin's bounding box (array size is `DIM`)
 *   localState   State of random number generator
 *
 * Output:
 *   x        Random values inside bin bounds (array size is `DIM`)
 */
__device__ void k_random_fn(double *x, const double *bin_min, const double *bin_max, util_curand_state_t *localState) {
  const int dim = DIM;
  int dim_id;

  for (dim_id=0; dim_id<dim; dim_id++) {
    x[dim_id] = (bin_max[dim_id] - bin_min[dim_id]) * curand_uniform_double(localState) + bin_min[dim_id];
  }
}

/**
 * Performs Monte Carlo integration in multi-dimensions of the function
 * `integrand_fn`.  The approximation of the integrand is computed along with
 * an approximated variance, which is interpreted as the (square of the)
 * integration error.
 *
 * Input:
 *   n_mc_evals   Number of Monte Carlo evaluations (i.e., samples)
 *   n_bin_evals  Number of Monte Carlo evaluations per bin
 *   bin_ranges   Boundaries between bins (array size: `dim*(N_BINS_PER_DIM+1)`)
 *   globalState  States of random number generators (array size is `n_blocks*m_threads`)
 *
 * Output:
 *   bin_content Values of integration for each bin (array size: `N_BINS_PER_DIM^DIM`)
 */
__global__ void k_mc_int_epoch(double *bin_content,
                               unsigned long n_mc_evals, unsigned long n_bin_evals, const double *bin_ranges,
                               util_curand_state_t *globalState) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  util_curand_state_t localState;
  const int dim = DIM;
  const int n_bins_per_dim = N_BINS_PER_DIM;
  const int n_bins_total   = pow(n_bins_per_dim, dim);
  double f = 0;
  int coord[DIM];
  double bin_min[DIM], bin_max[DIM];
  //int bin_id = 0;
  double x[DIM];
  // TODO (most code is missing but comments are placed for orientation)

  if (tid < n_bins_total) {
    /* copy state to local memory for efficiency */
    localState = globalState[tid];

    /* get bin parameters */
    get_bin_coordinates_from_index(coord, tid);
    get_bin_min_and_max(bin_min, bin_max, bin_ranges, coord);
    double bin_volume = get_bin_volume(bin_min, bin_max);

    /* evaluate/sample integrand */
    for (int j=0; j<n_bin_evals; j++) {
    k_random_fn(x, bin_min, bin_max, &localState);
    f += integrand_fn(x);
    }

    /* store intermediate result */
    // TODO set output bin_content[tid]
    bin_content[tid] = (n_bins_total * bin_volume * f);
    //printf("%d\n", localState);
    // localState += n_bins_total*bin_volume*f;

    /* copy state back to global memory */
    globalState[tid] = localState;
  }
}

/******************** GPU version ********************/

/**
 * Main function.
 */
int main(int argc, char **argv) {
  /* domain variables */
  const int dim = DIM;
  const double domain_min = DOMAIN_MIN;
  const double domain_max = DOMAIN_MAX;
  /* Monte Carlo variables */
  const int n_bins_per_dim = N_BINS_PER_DIM;
  const int n_bins_total   = pow(n_bins_per_dim, dim);
  const int n_bins_ranges_per_dim = N_BINS_PER_DIM + 1;
  const int n_bins_ranges_total   = dim*n_bins_ranges_per_dim;
  unsigned long n_mc_evals, n_mc_epochs, n_bin_evals;
  /* timing */
  clock_t time_begin, time_end;
  double  h_elapsed_s, d_elapsed_s;
  /* host memory */
  double  *h_bin_ranges;
  double  *h_bin_content;

  /* setup the number of Monte Carlo function evaluations from the command line */
  if (1 < argc) {
    n_mc_evals = atol(argv[1]);
  }
  else {
    printf("Argument missing for the number of Monte Carlo function evaluations.\n");
    printf("Run this program with a number for N:\n  ./mc_int N\n");
    exit(1);
  }
  if (2 < argc) {
    n_mc_epochs = atol(argv[2]);
  }
  else {
    n_mc_epochs = 1;
  }
  n_bin_evals = (n_mc_evals/n_mc_epochs) / n_bins_total;
  n_mc_evals  = n_mc_epochs * n_bin_evals * n_bins_total; // recalculate total number of evaluations

  printf("========================================\n");
  printf("  dim = %d\n", dim);
  printf("  n_bins_per_dim = %d, n_bins_total = %d\n", n_bins_per_dim, n_bins_total);
  printf("  n_mc_evals = %lu, n_mc_epochs = %lu, n_bin_evals = %lu\n", n_mc_evals, n_mc_epochs, n_bin_evals);
  printf("----------------------------------------\n");
  fflush(stdout); // flush buffered print outputs

  /* test conversion functions between bin indices and coodinates */
  int success;
  success = util_test_conversion_between_index_and_coord(dim, n_bins_per_dim,
                                                         get_bin_coordinates_from_index,
                                                         get_bin_index_from_coordinates);
  if (!success)  exit(1);

  /* allocate arrays in host memory (CPU) */
  h_bin_ranges  = (double*) malloc(n_bins_ranges_total*sizeof(double));
  h_bin_content = (double*) malloc(n_bins_total*sizeof(double));

  /* set bin ranges in all dimensions */
  for (int dim_id=0; dim_id<dim; dim_id++) {
    int offset = get_offset_for_bin_range_1dim(dim_id);
    setup_variable_bin_ranges_per_dim(&h_bin_ranges[offset], n_bins_per_dim, domain_min, domain_max);
  }
  /* print bin ranges */
  if (n_bins_per_dim <= 10) { // if not too many bins (say up to 10)
    print_bin_ranges(h_bin_ranges, dim, n_bins_per_dim);
  }

  /* get reference integral */
  double int_ref, err_ref;
  integral_reference(&int_ref, &err_ref);

  /******************** CPU version ********************/

  /* begin timing CPU version */
  time_begin = clock();

  /* run MC integration (CPU) */
  double h_int_approx, h_var_approx;
  // TODO uncomment when the function is implemented
  
  mc_int_epoch_sequential(&h_int_approx, &h_var_approx, h_bin_content,
                          n_mc_evals, n_bin_evals, h_bin_ranges);
  

  /* end timing CPU version */
  time_end = clock();
  h_elapsed_s = (time_end - time_begin) / ((double) CLOCKS_PER_SEC);

  /* compare integration error */
  printf("mc_int_epoch_sequential: integral ~ %.16f, std ~ %.3e\n", h_int_approx, sqrt(h_var_approx));
  printf("                         abs err = %.3e, rel err = %.3e\n", fabs(h_int_approx - int_ref), fabs(h_int_approx - int_ref)/fabs(int_ref));
  printf("                         wall-clock time [sec] ~ %g\n", h_elapsed_s);

  /******************** CPU version ********************/

  /******************** GPU version ********************/

  /* device memory */
  double  *d_bin_ranges;
  double  *d_bin_content;
  /* CUDA variables */
  const int n_threads = N_THREADS_PER_BLOCK;
  // TODO set the actual number of blocks, when the total amount of threads needs to be at least = n_bins_total
  const int n_blocks  = 8;
  util_curand_state_t *d_curand_states;

  printf("========================================\n");
  printf("  n_blocks = %d, n_threads = %d\n", n_blocks, n_threads);
  printf("----------------------------------------\n");
  fflush(stdout); // flush buffered print outputs

  /* create pseudo-random number generator */
  CUDA_CHKERR( cudaMalloc(&d_curand_states, n_blocks*n_threads*sizeof(util_curand_state_t)) );
  k_curand_setup_states<<< n_blocks, n_threads >>>(d_curand_states, 1234);
  CUDA_CHKERR(cudaGetLastError());

  /* allocate arrays in device memory (GPU) */
  // TODO
  cudaMalloc(&d_bin_ranges, n_bins_ranges_total*sizeof(double));
  cudaMalloc(&d_bin_content, n_bins_total*sizeof(double));

  /* begin timing GPU version */
  time_begin = clock();

  /* transfer data from host to device memory (CPU->GPU) */
  // TODO
  cudaMemcpy(d_bin_content, h_bin_content, n_bins_total*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bin_ranges, h_bin_ranges, n_bins_ranges_total*sizeof(double), cudaMemcpyHostToDevice);

  /* run partial MC integration (GPU) */
  // TODO uncomment when the kernel function is implemented
  k_mc_int_epoch<<< n_blocks, n_threads >>>(d_bin_content, n_mc_evals, n_bin_evals, d_bin_ranges, d_curand_states);
  CUDA_CHKERR(cudaGetLastError());

  /* wait for GPU threads to complete */
  CUDA_CHKERR( cudaDeviceSynchronize() );

  /* transfer data from device to host memory (GPU->CPU) */
  // TODO
  cudaMemcpy(h_bin_ranges, d_bin_ranges, n_bins_ranges_total*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bin_content, d_bin_content, n_bins_total*sizeof(double), cudaMemcpyDeviceToHost);

  /* reduce bin content (CPU) */
  double d_int_approx;
  reduce_bin_content_to_integral(&d_int_approx, h_bin_content, n_mc_evals);

  /* end timing GPU version */
  time_end = clock();
  d_elapsed_s = (time_end - time_begin) / ((double) CLOCKS_PER_SEC);

  /* compare integration error */
  printf("k_mc_int_epoch:          integral ~ %.16f\n", d_int_approx);
  printf("                         abs err = %.3e, rel err = %.3e\n", fabs(d_int_approx - int_ref), fabs(d_int_approx - int_ref)/fabs(int_ref));
  printf("                         wall-clock time [sec] ~ %g\n", d_elapsed_s);

  /* deallocate arrays in device memory (GPU) */
  // TODO
  cudaFree(d_bin_content);
  cudaFree(d_bin_ranges);

  /* destroy pseudo-random number generator */
  CUDA_CHKERR( cudaFree(d_curand_states) );

  /******************** GPU version ********************/

  /* deallocate arrays in host memory (CPU) */
  free(h_bin_ranges);
  free(h_bin_content);

  /* calculate speedups */
  double speedup_mc_int_epoch_sequential_vs_k_mc_int_epoch = h_elapsed_s / d_elapsed_s;
  printf("----------------------------------------\n");
  printf("Speedup, mc_int_epoch_sequential vs. k_mc_int_epoch: %.3e\n", speedup_mc_int_epoch_sequential_vs_k_mc_int_epoch);
  printf("========================================\n");

  return 0;
}
