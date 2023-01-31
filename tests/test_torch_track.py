import torch

from bmadx.bmad_torch.track_torch import (
    Beam,
    TorchCrabCavity,
    TorchDrift,
    TorchLattice,
    TorchQuadrupole,
)
from bmadx.bmad_torch.utils import get_transport_matrix


def test_lattice():
    q1 = TorchQuadrupole(torch.tensor(0.1), torch.tensor(1.0))
    d1 = TorchDrift(torch.tensor(1.0))
    c1 = TorchCrabCavity(torch.tensor(1.0), torch.tensor(100.0), torch.tensor(1.0))

    lattice = TorchLattice([q1, d1, c1])

    test_beam = Beam(
        torch.ones(6, 6),
        p0c=10.0e6,
    )
    print(lattice(test_beam))


def test_jacobian():
    ks = [
        torch.tensor(1.0),
        torch.linspace(0, 1, 10),
        torch.linspace(0, 1, 10).unsqueeze(-1),
    ]
    for k in ks:
        q1 = TorchQuadrupole(torch.tensor(0.1), k)
        assert q1.batch_shape == k.shape
        d1 = TorchDrift(torch.tensor(0.5))

        lattice = TorchLattice([q1, d1])
        assert lattice.batch_shape == k.shape

        J = get_transport_matrix(
            lattice, torch.tensor(0.0), torch.tensor(10.0e6), torch.tensor(0.511e6)
        )
        assert J.shape == torch.Size([*k.shape, 6, 6])



test_lattice()
test_jacobian()
