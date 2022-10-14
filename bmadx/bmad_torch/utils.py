import torch
from torch.autograd.functional import jacobian

from bmadx.track import Particle


def get_transport_matrix(lattice, s, p0c, mc2):
    def f(x):
        return lattice(Particle(*x, s, p0c, mc2))[:6]

    J = jacobian(f, torch.zeros(6))
    matrix_elements = []
    for ele in J[:6]:
        if lattice.batch_shape == torch.Size():
            matrix_elements += [ele.unsqueeze(0)]
        else:
            if ele.shape == torch.Size([*lattice.batch_shape, 6]):
                matrix_elements += [ele.unsqueeze(-2)]
            elif ele.shape == torch.Size([6]):
                matrix_elements += [ele.repeat(*lattice.batch_shape, 1, 1)]
            else:
                raise RuntimeError("unhandled shape for jacobian")

    return torch.cat(matrix_elements, dim=-2)
