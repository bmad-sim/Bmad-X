import yaml

import torch

import distgen
from distgen import Generator
from distgen.physical_constants import unit_registry as unit

from bmadx.constants import M_ELECTRON
from bmadx.bmad_torch.track_torch import par_to_beam
from bmadx.pmd_utils import openpmd_to_bmadx_particles

def create_pmd_particlegroup(
        base_yaml,
        transforms_yaml,
        n_particle
):
    """Creates openPMD ParticleGroup from dist and transform yaml files
    using distgen

        Params
        ------ 
        base_yaml: yaml file with base distribution parameters
        transform_yaml: yaml file with transforms
        
        Returns
        -------
        generated openPMD ParticleGroup
    """

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


def create_particles(
        base_yaml, 
        transforms_yaml, 
        p0c, 
        s = 0.0, 
        mc2 = M_ELECTRON,
        n_particle = None
):
    """
    Creates bmadx Particle distribution from dist and transform yaml files
    using distgen

        Params
        ------ 
        base_yaml: yaml file with base distribution parameters
        transform_yaml: yaml file with transforms
        p0c: reference momentum of Beam coordinates in eV
        save_as (str): if provided, saves the generated Beam

        Returns
        -------
        generated Bmad-X Particle namedtuple. 
    """

    pmd_par = create_pmd_particlegroup(
        base_yaml,
        transforms_yaml,
        n_particle
    )

    par = openpmd_to_bmadx_particles(
        pmd_par,
        p0c,
        s,
        mc2
    )

    return par

def create_beam(
        base_yaml, 
        transforms_yaml, 
        p0c, 
        s = torch.tensor(0.0, dtype=torch.float32), 
        mc2 = torch.tensor(M_ELECTRON, dtype=torch.float32),
        n_particle = None, 
        save_as = None
):
    """
    Creates bmadx torch Beam from dist and transform yaml files
    using distgen

        Params
        ------
        base_yaml: yaml file with base distribution parameters
        transform_yaml: yaml file with transforms

        p0c: reference momentum of Beam coordinates in eV
        save_as (str): if provided, saves the generated Beam
        
        Returns
        -------
        generated Bmad-X torch Beam. 
    """

    par = create_particles(
        base_yaml,
        transforms_yaml,
        p0c,
        s,
        mc2,
        n_particle
    )

    beam = par_to_beam(par)

    if save_as is not None:
        torch.save(beam, save_as)
        print(f'ground truth distribution saved at {save_as}')

    return beam


"""
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
"""
