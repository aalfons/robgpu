void colwisemean(const MYTYPE * x, size_t nCols, size_t nRows, MYTYPE * means);
void colwisemean_internal(MYTYPE * d_x, size_t nCols, size_t nRows, MYTYPE * d_means);
// kernel to be optimized to divide each element by value
__global__ void divide_by_value_kernel(MYTYPE * d_x, size_t n, size_t value);

