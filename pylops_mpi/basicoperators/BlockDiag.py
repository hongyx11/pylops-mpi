import numpy as np
from scipy.sparse.linalg._interface import _get_dtype
from mpi4py import MPI
from typing import Optional, Sequence

from pylops.utils import DTypeLike
from pylops import LinearOperator

from pylops_mpi import DistributedArray


class MPIBlockDiag(LinearOperator):
    r"""MPI Block-diagonal operator.

        Create a block-diagonal operator from N linear operators using MPI.

        Parameters
        ----------
        ops : :obj:`list`
            Linear operators to be stacked. Alternatively,
            :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrices can be passed
            in place of one or more operators.
        base_comm : :obj:`mpi4py.MPI.Comm`, optional
            Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
        dtype : :obj:`str`, optional
            Type of elements in input array.

        Attributes
        ----------
        shape : :obj:`tuple`
            Operator shape
        explicit : :obj:`bool`
            Operator contains a matrix that can be solved explicitly (``True``) or
            not (``False``)

        Notes
        -----
        Refer to :obj:pylops.BlockDiag for more details.
    """

    def __init__(self, ops: Sequence[LinearOperator],
                 base_comm: MPI.Comm = MPI.COMM_WORLD,
                 dtype: Optional[DTypeLike] = None):
        # Required for MPI
        self.base_comm = base_comm
        self.rank = base_comm.Get_rank()
        self.size = base_comm.Get_size()
        self.ops = ops
        mops = np.zeros(len(self.ops), dtype=np.int64)
        nops = np.zeros(len(self.ops), dtype=np.int64)
        for iop, oper in enumerate(self.ops):
            nops[iop] = oper.shape[0]
            mops[iop] = oper.shape[1]
        self.mops = mops.sum()
        self.nops = nops.sum()
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        # Shape of the operator is equal to the sum of nops and mops at all ranks
        self.shape = (self.base_comm.allreduce(self.nops), self.base_comm.allreduce(self.mops))
        # Shape of the operator at each rank
        self.localop_shape = (self.nops, self.mops)
        if dtype:
            self.dtype = _get_dtype(dtype)
        else:
            self.dtype = np.dtype(dtype)
        clinear = all([getattr(oper, "clinear", True) for oper in self.ops])
        super().__init__(shape=self.shape, dtype=self.dtype, clinear=clinear)

    def _matvec(self, x: DistributedArray) -> DistributedArray:
        if x.local_shape != (self.mops, ):
            raise ValueError(f"Dimension mismatch: x shape-{x.local_shape} does not match operator shape "
                             f"{self.localop_shape}; {x.local_shape[0]} != {self.mops} (dim1) at rank={self.rank}")
        y = DistributedArray(global_shape=self.shape[0], dtype=x.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.matvec(x.local_array[self.mmops[iop]:
                                                self.mmops[iop + 1]]))
        if y1:
            y[:] = np.concatenate(y1)
        return y

    def _rmatvec(self, x: DistributedArray) -> DistributedArray:
        if x.local_shape != (self.nops, ):
            raise ValueError(f"Dimension mismatch: x shape-{x.local_shape} does not match operator shape "
                             f"{self.localop_shape}; {x.local_shape[0]} != {self.nops} (dim0) at rank={self.rank}")
        y = DistributedArray(global_shape=self.shape[1], dtype=x.dtype)
        y1 = []
        for iop, oper in enumerate(self.ops):
            y1.append(oper.rmatvec(x.local_array[self.nnops[iop]:
                                                 self.nnops[iop + 1]]))
        if y1:
            y[:] = np.concatenate(y1)
        return y
