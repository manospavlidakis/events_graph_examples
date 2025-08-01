#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d (%d): %s\n", __FILE__, __LINE__,    \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void kernel1(unsigned long long clock_count, int *x) {
  unsigned long long start = clock64();
  printf("Kernel 1 running on stream1\n");
  while (clock64() - start < clock_count);
  printf("Kernel 1 (after busy wait): x=%d\n", *x);
  *x += 1;
}

__global__ void kernel2(int *x) { 
    printf("Kernel 2 (before exec): x=%d\n", *x);
    *x = 0; 
  }

int main(int argc, char **argv) {
  bool useExternalWait = false;

  // Check for optional flag
  if (argc > 1 && strcmp(argv[1], "--use-wait") == 0) {
    useExternalWait = true;
    std::cout << "INFO: External wait ENABLED via command-line argument.\n";
  } else {
    std::cout << "INFO: External wait DISABLED (default behavior).\n";
  }
  cudaStream_t stream1, stream2;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

  const unsigned long long clocks = 5e9; // 1e7=~6.7 ms, 1e8=~67 ms, 5e9=~3.3 sec
  int *d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, sizeof(int)));
  CHECK_CUDA(cudaMemset(d, -1, sizeof(int)));

  cudaEvent_t extEvent;
  CHECK_CUDA(cudaEventCreateWithFlags(&extEvent, cudaEventDisableTiming));
 
  // START capture 1
  cudaGraph_t graph;
  CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
  kernel1<<<1, 1, 0, stream1>>>(clocks, d);
  CHECK_CUDA(cudaEventRecordWithFlags(extEvent, stream1, cudaEventRecordExternal));
  CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));
  // END capture 1

  // START capture 2
  cudaGraph_t graph2;
  CHECK_CUDA(cudaStreamBeginCapture(stream2, cudaStreamCaptureModeGlobal));
  // Wait for kernel 1 to finish
  if (useExternalWait) {
    CHECK_CUDA(cudaStreamWaitEvent(stream2, extEvent, cudaEventWaitExternal));
  }
  kernel2<<<1, 1, 0, stream2>>>(d);
  CHECK_CUDA(cudaStreamEndCapture(stream2, &graph2));
  // END capture 2

  // Instantiate and launch graph
  cudaGraphExec_t graphExec, graphExec2;
  CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  CHECK_CUDA(cudaGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));

  CHECK_CUDA(cudaGraphLaunch(graphExec, stream1));
  CHECK_CUDA(cudaGraphLaunch(graphExec2, stream2));

  CHECK_CUDA(cudaStreamSynchronize(stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream2));

  // Cleanup
  CHECK_CUDA(cudaGraphExecDestroy(graphExec));
  CHECK_CUDA(cudaGraphExecDestroy(graphExec2));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaGraphDestroy(graph2));
  CHECK_CUDA(cudaEventDestroy(extEvent));
  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream2));

  return 0;
}


/**************************************
Without wait
./simple_event_record_flags_external_wait_ext_two_captures
INFO: External wait DISABLED (default behavior).
Kernel 1 running on stream1 // Kernel 1 starts 
Kernel 2 (before exec): x=-1 // Due to busy wait of kernel 1, kernel 2 overcomes 
// it, so it sets x to 0. !! x is different to 0 because it is before x=0
Kernel 1 (after busy wait): x=0 // After busy wait kernel 1 sees that x is 0

With wait
./simple_event_record_flags_external_wait_ext_two_captures --use-wait
INFO: External wait ENABLED via command-line argument.
Kernel 1 running on stream1
Kernel 1 (after busy wait): x=-1
Kernel 2 (before exec): x=-1
*****************************************/