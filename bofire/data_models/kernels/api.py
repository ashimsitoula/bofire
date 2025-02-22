from typing import Union

from bofire.data_models.kernels.aggregation import (
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
)
from bofire.data_models.kernels.categorical import (
    CategoricalKernel,
    HammondDistanceKernel,
)
from bofire.data_models.kernels.continuous import (
    ContinuousKernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
)
from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.kernels.molecular import MolecularKernel, TanimotoKernel

AbstractKernel = Union[Kernel, CategoricalKernel, ContinuousKernel, MolecularKernel]

AnyContinuousKernel = Union[
    MaternKernel,
    LinearKernel,
    RBFKernel,
]

AnyCategoricalKernal = HammondDistanceKernel

AnyMolecularKernel = TanimotoKernel

AnyKernel = Union[
    AdditiveKernel,
    MultiplicativeKernel,
    ScaleKernel,
    HammondDistanceKernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
    TanimotoKernel,
]
