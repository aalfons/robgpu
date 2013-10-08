void checkCudaError(const char * msg);
int hasCudaError(const char * msg);
float * getMatFromFile(int rows, int cols, const char * fn);
void checkCublasError(const char * msg);
int hasCublasError(const char * msg);
