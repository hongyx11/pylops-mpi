# save as mpi_cupy_example.py
from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("rank + ", rank)
N = 10
if size < 2:
    if rank == 0:
        print("Run with at least 2 ranks.")
    raise SystemExit(1)

if rank == 0:
    # GPU array on rank 0
    a = cp.arange(N, dtype=cp.float64)
    # make sure GPU work is done before MPI call
    cp.cuda.Stream.null.synchronize()
    # send device array directly (CUDA-aware MPI will transfer from device)
    comm.Send([a, MPI.DOUBLE], dest=1, tag=77)
elif rank == 1:
    b = cp.empty(N, dtype=cp.float64)
    # receive into device memory directly
    comm.Recv([b, MPI.DOUBLE], source=0, tag=77)
    # sync and check
    cp.cuda.Stream.null.synchronize()
    print("rank 1 got:", b)

