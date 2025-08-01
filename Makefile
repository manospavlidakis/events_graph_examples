# Makefile to build all CUDA event sync examples

NVCC := nvcc
NVCC_FLAGS := -std=c++17 -arch=sm_86

# List of source files
SOURCES := \
    record_wait_with_flags_1graph.cu \
    record_wait_with_flags_2graphs.cu \
    simple_event_record.cu


# Convert .cu to binary names
BINARIES := $(SOURCES:.cu=)

# Default rule
all: $(BINARIES)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(BINARIES)

.PHONY: all clean

