#from scalene import scalene_profiler

import sys
#print(sys.argv)
n_particles = int(sys.argv[1])
n_quadrupoles = int(sys.argv[2])
n_slices = int(sys.argv[3])

import torch

from bmadx import Particle, Drift, Quadrupole, track_lattice, M_ELECTRON

def create_gaussian_beam(n_particles, mean, cov, s, p0c, mc2):
    torch.manual_seed(0)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    coords = dist.sample(torch.Size([n_particles])) # particles' coordinates
    
    return Particle(*coords.T, s, p0c, mc2)

class BeamlineModel(torch.nn.Module):
    def __init__(self, k_set, l_d, l_q, beam_in, sigma_t):
        super().__init__()
        self.register_parameter('k_set', torch.nn.Parameter(k_set))
        self.l_d = l_d
        self.l_q = l_q
        self.beam_in = beam_in
        self.sigma_t = sigma_t
    def forward(self):
        # create lattice
        lattice = []
        half_drift = Drift( L = self.l_d/2)
        for k in self.k_set:
            lattice.append( half_drift )
            lattice.append( Quadrupole(L = self.l_q,
                                       K1 = k,
                                       NUM_STEPS = n_slices) )
            lattice.append( half_drift )
            
        beam_out = track_lattice(self.beam_in, lattice)
        dx = beam_out.x.std() - self.sigma_t
        dy = beam_out.y.std() - self.sigma_t
        
        return torch.sqrt( dx**2 + dy**2 )

# model constants
l_d = 0.9 # drift length
l_q = 0.1 # quad length
sigma_t = 5e-3 # target beam size

# incoming beam
beam_in = create_gaussian_beam(n_particles, 
                               mean = torch.zeros(6)*1e-3, 
                               cov = torch.eye(6)*1e-6, 
                               s = torch.tensor(0.0), 
                               p0c = torch.tensor(4e7),
                               mc2 = torch.tensor(M_ELECTRON))

# model evaluation and backward pass
k_set = torch.zeros(n_quadrupoles, requires_grad=True)
#scalene_profiler.start()
model = BeamlineModel(k_set, l_d, l_q, beam_in, sigma_t)
loss = model()
loss.backward()
#scalene_profiler.stop()