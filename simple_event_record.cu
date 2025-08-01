#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d (%d): %s\n", __FILE__, __LINE__,    \
              err, cudaGetErrorString(err));                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void kernelA() { printf("kernelA running on stream1\n"); }

__global__ void kernelB() {
  printf("kernelB running on stream2 AFTER kernelA\n");
}

int main() {
  cudaStream_t stream1, stream2;
  CHECK_CUDA(cudaStreamCreate(&stream1));
  CHECK_CUDA(cudaStreamCreate(&stream2));

  cudaEvent_t extEvent;
  CHECK_CUDA(cudaEventCreateWithFlags(&extEvent, cudaEventDisableTiming));

  cudaGraph_t graph;
  CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

  // kernelA on stream1
  kernelA<<<1, 1, 0, stream1>>>();

  CHECK_CUDA(
       cudaEventRecord(extEvent, stream1));
  // stream2 waits on extEvent (from stream1) inside capture
  CHECK_CUDA(cudaStreamWaitEvent(stream1, extEvent));

  // kernelB on stream2
  kernelB<<<1, 1, 0, stream2>>>();

  CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));

  // Instantiate and launch graph
  cudaGraphExec_t graphExec;
  CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  CHECK_CUDA(cudaGraphLaunch(graphExec, stream1));
  CHECK_CUDA(cudaStreamSynchronize(stream1));

  // Cleanup
  CHECK_CUDA(cudaGraphExecDestroy(graphExec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaEventDestroy(extEvent));
  CHECK_CUDA(cudaStreamDestroy(stream1));
  CHECK_CUDA(cudaStreamDestroy(stream2));

  return 0;
}
