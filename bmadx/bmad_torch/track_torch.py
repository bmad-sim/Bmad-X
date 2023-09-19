import torch
import copy
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from bmadx.structures import Particle
from bmadx.constants import M_ELECTRON
from bmadx import LIB_DICT


class Beam(torch.nn.Module):
    def __init__(self, data, p0c, s=torch.tensor(0.0), mc2=torch.tensor(M_ELECTRON)):
        super(Beam, self).__init__()
        self.keys = ["x", "px", "y", "py", "z", "pz"]

        self.register_buffer("data", data)
        self.register_parameter("p0c", Parameter(p0c, requires_grad=False))
        self.register_parameter("s", Parameter(s, requires_grad=False))
        self.register_parameter("mc2", Parameter(mc2, requires_grad=False))

    @property
    def x(self):
        return self.data[..., 0]

    @property
    def px(self):
        return self.data[..., 1]

    @property
    def y(self):
        return self.data[..., 2]

    @property
    def py(self):
        return self.data[..., 3]

    @property
    def z(self):
        return self.data[..., 4]

    @property
    def pz(self):
        return self.data[..., 5]

    def to_list_of_beams(self):
        beams = []
        for i in range(len(getattr(self, self.keys[0]))):
            beams += [
                Particle(
                    *[getattr(self, key)[i] for key in self.keys], **self._defaults
                )
            ]

        return beams
    
    def detach_clone(self):
        return Beam(self.data.detach().clone(), self.p0c, self.s, self.mc2)
    
    def numpy_particles(self):
        return Particle(*self.data.detach().clone().numpy().T,
                        p0c = self.p0c.detach().clone().numpy(),
                        s = self.s.detach().clone().numpy(),
                        mc2 = self.mc2.detach().clone().numpy())


class TorchElement(Module):
    def __init__(self, tracking_function):
        super(TorchElement, self).__init__()
        self.track = tracking_function

    def forward(self, beam):
        #par = self.track(beam, self)
        #return particle_to_beam(par)
        return self.track(beam, self)

    @property
    def batch_shape(self):
        out = torch.broadcast_tensors(*[val.data for _, val in self.named_parameters()])
        return out[0].shape
    
class TorchDrift(TorchElement):
    def __init__(
        self,
        L: Tensor
    ):
        super(TorchDrift, self).__init__(
            LIB_DICT[torch]['tracking_routine']['Drift']
        )
        self.register_parameter("L", Parameter(L, requires_grad=False))
        

class TorchQuadrupole(TorchElement):
    def __init__(
        self,
        L: Tensor,
        K1: Tensor,
        NUM_STEPS: int = 1,
        X_OFFSET: Tensor = torch.tensor(0.0),
        Y_OFFSET: Tensor = torch.tensor(0.0),
        TILT: Tensor = torch.tensor(0.0)
    ):
        super(TorchQuadrupole, self).__init__(
            LIB_DICT[torch]['tracking_routine']['Quadrupole']
        )
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_parameter(
            "NUM_STEPS", Parameter(torch.tensor(NUM_STEPS), requires_grad=False)
        )
        self.register_parameter("TILT", Parameter(TILT, requires_grad=False))
        self.register_parameter("K1", Parameter(K1, requires_grad=False))

class TorchSextupole(TorchElement):
    def __init__(
        self,
        L: Tensor,
        K2: Tensor,
        NUM_STEPS: int = 1,
        X_OFFSET: Tensor = torch.tensor(0.0),
        Y_OFFSET: Tensor = torch.tensor(0.0),
        TILT: Tensor = torch.tensor(0.0),
    ):
        super(TorchSextupole, self).__init__(
            LIB_DICT[torch]['tracking_routine']['Sextupole']
        )
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_parameter(
            "NUM_STEPS", Parameter(torch.tensor(NUM_STEPS), requires_grad=False)
        )
        self.register_parameter("TILT", Parameter(TILT, requires_grad=False))
        self.register_parameter("K2", Parameter(K2, requires_grad=False))

class TorchCrabCavity(TorchElement):
    def __init__(
        self,
        L: Tensor,
        VOLTAGE: Tensor,
        RF_FREQUENCY: Tensor,
        PHI0: Tensor = torch.tensor(0.0),
        X_OFFSET: Tensor = torch.tensor(0.0),
        Y_OFFSET: Tensor = torch.tensor(0.0),
        TILT: Tensor = torch.tensor(0.0)
    ):
        super(TorchCrabCavity, self).__init__(
            LIB_DICT[torch]['tracking_routine']['CrabCavity']
        )
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("VOLTAGE", Parameter(VOLTAGE, requires_grad=False))
        self.register_parameter("PHI0", Parameter(PHI0, requires_grad=False))
        self.register_parameter(
            "RF_FREQUENCY", Parameter(RF_FREQUENCY, requires_grad=False)
        )
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_parameter("TILT", Parameter(TILT, requires_grad=False))
        
        
class TorchRFCavity(TorchElement):
    def __init__(
        self,
        L: Tensor,
        VOLTAGE: Tensor,
        RF_FREQUENCY: Tensor,
        PHI0: Tensor = torch.tensor(0.0),
        X_OFFSET: Tensor = torch.tensor(0.0),
        Y_OFFSET: Tensor = torch.tensor(0.0),
        TILT: Tensor = torch.tensor(0.0)
    ):
        super(TorchRFCavity, self).__init__(
            LIB_DICT[torch]['tracking_routine']['RFCavity']
        )
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("VOLTAGE", Parameter(VOLTAGE, requires_grad=False))
        self.register_parameter("PHI0", Parameter(PHI0, requires_grad=False))
        self.register_parameter(
            "RF_FREQUENCY", Parameter(RF_FREQUENCY, requires_grad=False)
        )
        self.register_parameter("X_OFFSET", Parameter(X_OFFSET, requires_grad=False))
        self.register_parameter("Y_OFFSET", Parameter(Y_OFFSET, requires_grad=False))
        self.register_parameter("TILT", Parameter(TILT, requires_grad=False))
        
        
class TorchSBend(TorchElement):
    def __init__(
        self,
        L: Tensor,
        P0C: Tensor,
        G: Tensor,
        DG: Tensor = torch.tensor(0.0),
        E1: Tensor = torch.tensor(0.0),
        E2: Tensor = torch.tensor(0.0),
        F_INT: Tensor = torch.tensor(0.0),
        H_GAP: Tensor = torch.tensor(0.0),
        F_INT_X: Tensor = torch.tensor(0.0),
        H_GAP_X: Tensor = torch.tensor(0.0),
        FRINGE_AT: str = "both_ends",
        FRINGE_TYPE: str = "none"
    ):
        
        fringe_at_dic =  {
            "no_end": 0,
            "both_ends": 1,
            "entrance_end": 2,
            "exit_end":3
        }
        fringe_type_dic =  {
            "none": 0,
            "soft_edge_only": 1,
            "hard_edge_only": 2,
            "full": 3,
            "linear_edge": 4,
            "basic_bend": 5
        }
        inv_fringe_at_dic = dict(map(reversed, fringe_at_dic.items()))
        inv_fringe_type_dic = dict(map(reversed, fringe_type_dic.items()))
        
        super(TorchSBend, self).__init__(
            LIB_DICT[torch]['tracking_routine']['SBend']
        )
        self.register_parameter("L", Parameter(L, requires_grad=False))
        self.register_parameter("P0C", Parameter(P0C, requires_grad=False))
        self.register_parameter("G", Parameter(G, requires_grad=False))
        self.register_parameter("DG", Parameter(DG, requires_grad=False))
        self.register_parameter("E1", Parameter(E1, requires_grad=False))
        self.register_parameter("E2", Parameter(E2, requires_grad=False))
        self.register_parameter("E1", Parameter(F_INT, requires_grad=False))
        self.register_parameter("E2", Parameter(H_GAP, requires_grad=False))
        self.register_parameter("F_INT", Parameter(F_INT_X, requires_grad=False))
        self.register_parameter("H_GAP", Parameter(H_GAP_X, requires_grad=False))
        self.register_parameter("F_INT_X", Parameter(F_INT_X, requires_grad=False))
        self.register_parameter("H_GAP_X", Parameter(H_GAP_X, requires_grad=False))
        self.FRINGE_AT = FRINGE_AT
        self.FRINGE_TYPE = FRINGE_TYPE
        #self.register_buffer("FRINGE_AT", torch.tensor(fringe_at_dic[FRINGE_AT]))
        #self.register_buffer("FRINGE_TYPE", torch.tensor(fringe_type_dic[FRINGE_TYPE]))
        
        

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
    
    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_elements(self):
        return len(self.elements)

    @property
    def batch_shape(self):
        out = torch.broadcast_tensors(
            *[torch.ones(ele.batch_shape) for ele in self.elements]
        )
        return out[0].shape

def particle_to_beam(particle: Particle):
    '''
    Returns Beam corresponding to Particle. 
    '''

    if type(particle.x) != torch.Tensor:
        n_par = len(particle.x)
        coords = torch.zeros((n_par, 6))
        for i in range(6):
            coords[:,i] = torch.tensor(particle[i])
    else:
        coords = torch.vstack(particle[:6]).T

    params = [0,0,0]
    for i in (0, 1, 2):
        if type(particle[6+i]) != torch.Tensor:
            params[i] = torch.tensor(particle[6+i])
        else:
            params[i] = particle[6+i]


    return Beam(
        coords,
        p0c = params[0],
        s = params[1],
        mc2 = params[2]
    )