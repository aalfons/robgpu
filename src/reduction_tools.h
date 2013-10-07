
#ifndef _robgpu_REDUCTION_TOOLS_H_
#define _robgpu_REDUCTION_TOOLS_H_

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

extern "C" bool isPow2(unsigned int x);
extern "C" unsigned int nextPow2(unsigned int x);

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

#endif // _robgpu_REDUCTION_TOOLS_H_ 

