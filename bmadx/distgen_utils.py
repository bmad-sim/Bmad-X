import yaml

import torch
import numpy as np

import distgen
from distgen import Generator
from distgen.physical_constants import unit_registry as unit

from bmadx.constants import M_ELECTRON
from bmadx.coordinates import openpmd_to_bmadx
from bmadx.bmad_torch.track_torch import Beam

def create_pmd_particlegroup(base_yaml, transforms_yaml, n_particle):
    '''Creates openPMD ParticleGroup from dist and transform yaml files
    using distgen

    Args: 
        base_yaml: yaml file with base distribution parameters
        transform_yaml: yaml file with transforms
    
    Returns:
        generated openPMD ParticleGroup
    '''

    gen = Generator(base_yaml)

    with open(transforms_yaml) as f:
        transforms_dict = yaml.safe_load(f)

    if distgen.__version__ >= '1.0.0':
        gen["transforms"] = transforms_dict
        if n_particle is not None:
            gen['n_particle'] = n_particle
    else:
        gen.input["transforms"] = transforms_dict
        if n_particle is not None:
            gen.input['n_particle'] = n_particle

    particle_group = gen.run()
    particle_group.drift_to_z(z=0)

    return particle_group

def create_beam(base_yaml, transforms_yaml, p0c, n_particle=None, save_as=None):
    '''Creates bmadx torch Beam from dist and transform yaml files
    using distgen

    Args: 
        base_yaml: yaml file with base distribution parameters
        transform_yaml: yaml file with transforms
        p0c: reference momentum of Beam coordinates in eV
        save_as (str): if provided, saves the generated Beam
    
    Returns:
        generated Bmad-X torch Beam. 
    '''

    # Generate openPMD particle group:
    particle_group = create_pmd_particlegroup(base_yaml, transforms_yaml, n_particle)

    # Transform to Bmad phase space coordinates:
    coords = np.array(openpmd_to_bmadx(particle_group, p0c)).T
    tkwargs = {"dtype": torch.float32}
    coords = torch.tensor(coords, **tkwargs)

    # create Bmad-X torch Beam:
    beam = Beam(
        coords,
        s=torch.tensor(0.0, **tkwargs),
        p0c=torch.tensor(p0c, **tkwargs),
        mc2=torch.tensor(M_ELECTRON, **tkwargs)
    )

    # save ground truth beam
    if save_as is not None:
        torch.save(beam, save_as)
        print(f'ground truth distribution saved at {save_as}')

    return beam
