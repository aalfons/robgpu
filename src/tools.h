__global__ void columnwisecenter_kernel(MYTYPE * d_x, size_t n, MYTYPE * d_col_means);

__global__ void subsample_kernel(MYTYPE * d_x, size_t n, MYTYPE * d_x_subsample, size_t sample_size, int * d_sample_index, int nsamp, int actual_sample);

