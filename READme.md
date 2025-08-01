The example record_wait_with_flags_2graphs has two options for execution one that uses cudaStreamWaitEvent and one that do not:
## Without wait
./record_wait_with_flags_2graphs
INFO: External wait DISABLED (default behavior).
Kernel 1 running on stream1 // Kernel 1 starts 
Kernel 2 (before exec): x=-1 // Due to busy wait of kernel 1, kernel 2 overcomes 
// it, so it sets x to 0. !! x is different to 0 because it is before x=0
Kernel 1 (after busy wait): x=0 // After busy wait kernel 1 sees that x is 0

## With wait
./record_wait_with_flags_2graphs --use-wait
INFO: External wait ENABLED via command-line argument.
Kernel 1 running on stream1
Kernel 1 (after busy wait): x=-1
Kernel 2 (before exec): x=-1
