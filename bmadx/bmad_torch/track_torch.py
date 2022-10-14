import torch
from torch import Tensor
from torch.autograd.functional import jacobian
from torch.nn import Module, ModuleList, Parameter

from bmadx import track


class Beam(torch.nn.Module):
    def __init__(self, data, p0c: float, s: float = 0.0, mc2: float = 0.511e6):
        super(Beam, self).__init__()
        self.keys = ["x", "px", "y", "py", "z", "pz"]
        self.data = data

        for i, key in enumerate(self.keys):
            self.register_parameter(key, Parameter(data[..., i], requires_grad=False))

        self.register_parameter(
            "p0c", Parameter(torch.tensor(p0c), requires_grad=False)
        )
        self.register_parameter("s", Parameter(torch.tensor(s), requires_grad=False))
        self.register_parameter(
            "mc2", Parameter(torch.tensor(mc2), requires_grad=False)
        )

    def to_list_of_beams(self):
        beams = []
        for i in range(len(getattr(self, self.keys[0]))):
            beams += [
                track.Particle(
                    *[getattr(self, key)[i] for key in self.keys], **self._defaults
                )
            ]

        return beams


class TorchElement(Module):
    def __init__(self, tracking_function):
        super(TorchElement, self).__init__()
        self.track = tracking_function(torch)

    def forward(self, X):
        return self.track(X, self)

    @property
    def batch_shape(self):
        out = torch.broadcast_tensors(*[val.data for _, val in self.named_parameters()])
        return out[0].shape


class TorchQuadrupole(TorchElement):
    def __init__(
        self,
        L: Tensor,
        K1: Tensor,
        NUM_STEPS: int = 1,
        X_OFFSET: Tensor = torch.tensor(0.0),
        Y_OFFSET: Tensor = torch.tensor(0.0),
        TILT: Tensor = torch.tensor(0.0),
    ):
        super(TorchQuadrupole, self).__init__(track.make_track_a_quadrupole)
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_parameter(
            "NUM_STEPS", Parameter(torch.tensor(NUM_STEPS), requires_grad=False)
        )
        self.register_parameter("TILT", Parameter(TILT, requires_grad=False))
        self.register_parameter("K1", Parameter(K1, requires_grad=False))


class TorchCrabCavity(TorchElement):
    def __init__(
        self,
        L: Tensor,
        VOLTAGE: Tensor,
        RF_FREQUENCY: Tensor,
        PHI0: Tensor = torch.tensor(0.0),
        X_OFFSET: Tensor = torch.tensor(0.0),
        Y_OFFSET: Tensor = torch.tensor(0.0),
        TILT: Tensor = torch.tensor(0.0),
    ):
        super(TorchCrabCavity, self).__init__(track.make_track_a_crab_cavity)
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_parameter("VOLTAGE", Parameter(VOLTAGE, requires_grad=False))
        self.register_parameter("TILT", Parameter(TILT, requires_grad=False))
        self.register_parameter("PHI0", Parameter(PHI0, requires_grad=False))
        self.register_parameter(
            "RF_FREQUENCY", Parameter(RF_FREQUENCY, requires_grad=False)
        )


class TorchDrift(TorchElement):
    def __init__(
        self,
        L: Tensor,
    ):
        super(TorchDrift, self).__init__(track.make_track_a_drift)
        self.register_parameter("L", Parameter(L, requires_grad=False))


class TorchLattice(Module):
    def __init__(self, elements, only_last=True):
        super(TorchLattice, self).__init__()
        self.elements = ModuleList(elements)
        self.only_last = only_last

    def forward(self, p_in):
        if self.only_last:
            p = p_in
            for i in range(self.n_elements):
                p = self.elements[i](p)

            return p
        else:
            all_p = [None] * (self.n_elements + 1)
            all_p[0] = p_in

            for i in range(self.n_elements):
                all_p[i + 1] = self.elements[i](all_p[i])

            return all_p

    @property
    def n_elements(self):
        return len(self.elements)

    @property
    def batch_shape(self):
        out = torch.broadcast_tensors(
            *[torch.ones(ele.batch_shape) for ele in self.elements]
        )
        return out[0].shape
