from bmadx.constants import C_LIGHT, M_ELECTRON, E_CHARGE
from bmadx.structures import Particle
from bmadx.bmad_torch.track_torch import Beam, par_to_beam
import numpy as np
import torch
import sys

from pmd_beamphysics import ParticleGroup


def openpmd_to_bmadx_coords(
        pmd_par: ParticleGroup,
        p0c
):
    """
    Transforms openPMD-beamphysics ParticleGroup to 
    bmad phase-space coordinates.

        Parameters:
            pmd_par (pmd_beamphysics.ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV

        Returns:
            bmad_coods (list): list of bmad coords (x, px, y, py, z, pz)
    """

    x = pmd_par.x
    px = pmd_par.px / p0c
    y = pmd_par.y
    py = pmd_par.py / p0c
    z = - pmd_par.beta * C_LIGHT * pmd_par.t
    pz = pmd_par.p / p0c - 1.0

    bmad_coords = (x, px, y, py, z, pz)

    return bmad_coords


def openpmd_to_bmadx_particles(
        pmd_par: ParticleGroup,
        p0c: float,
        s : float = 0.0,
        mc2 : float = M_ELECTRON
        ):
    """
    Transforms openPMD-beamphysics ParticleGroup to 
    bmad phase-space Particle named tuple.

        Parameters:
            pmd_par (pmd_beamphysics.ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV

        Returns:
            Bmadx Particle
    """
    coords = openpmd_to_bmadx_coords(pmd_par, p0c)
    par = Particle(
        *coords, 
        s = s,
        p0c = p0c,
        mc2 = mc2)
    return par


def openpmd_to_bmadx_beam(
        pmd_par: ParticleGroup,
        p0c,
        s = torch.tensor(0.0, dtype=torch.float32),
        mc2 = torch.tensor(M_ELECTRON, dtype=torch.float32)
        ):
    """
    Transforms openPMD-beamphysics ParticleGroup to 
    bmad phase-space Particle named tuple.

        Parameters:
            pmd_par (pmd_beamphysics.ParticleGroup): openPMD-beamphysics ParticleGroup
            p0c (float): reference momentum in eV

        Returns:
            Bmadx torch Beam
    """
    par = openpmd_to_bmadx_particles(pmd_par, p0c, s, mc2)
    beam = par_to_beam(par)
    return beam


def bmadx_particles_to_openpmd(par: Particle):
    """
    Transforms bmadx Particle to openPMD-beamphysics ParticleGroup.

        Parameters
        ----------
        par: bmax Particle
            particle to transform.

        Returns
        -------
        pmd_beamphysics.ParticleGroup
    """
    lib = sys.modules[type(par.x).__module__]
    if lib == np:
        x = par.x
        px = par.px
        y = par.y
        py = par.py
        z = par.z
        pz = par.pz
    elif lib == torch:
        x = par.x.detach().numpy()
        px = par.px.detach().numpy()
        y = par.y.detach().numpy()
        py = par.py.detach().numpy()
        z = par.z.detach().numpy()
        pz = par.pz.detach().numpy()
    else:
        raise ValueError('Only numpy and torch Particles are supported as of now')

    dat = {}

    dat['x'] = x
    dat['px'] = px * par.p0c
    dat['y'] = y
    dat['py'] = py * par.p0c
    dat['z'] = pz * 0.0
    dat['pz'] = par.p0c * ( (pz + 1.0)**2 - px**2 - py**2 )**0.5

    p = (1 + pz ) * par.p0c
    beta = ( 
        (p / M_ELECTRON)**2 / 
        ( 1 + (p / M_ELECTRON)**2 )
    )**0.5

    dat['t'] = - z / (C_LIGHT * beta)

    dat['status'] = np.ones_like(x, dtype=int)
    dat['weight'] = - np.ones_like(x) * E_CHARGE

    if np.isclose(par.mc2, M_ELECTRON):
        dat['species'] = 'electron'
    else:
        raise ValueError('only electrons are supported as of now')
    
    return ParticleGroup(data=dat)


def bmadx_beam_to_openpmd(beam: Beam):
    """
    Transforms bmadx torch Beam to openPMD-beamphysics ParticleGroup.

        Parameters
        ----------
        beam: bmax torch Beam to transform.

        Returns
        -------
        pmd_beamphysics.ParticleGroup
    """
    par = beam.numpy_particles()
    pmd_par = bmadx_particles_to_openpmd(par)
    return pmd_par


def save_particles_as_h5(par: Particle, fname: str):
    """
    Saves bmadx Particle as h5 file in openPMD-beamphysics
    ParticleGroup standard.

        Parameters
        ----------
        par: bmax Particle
            particle to transform.

        fname: str
            file name 

        Returns
        -------
        None
    """
    pmd_par = bmadx_particles_to_openpmd(par)
    pmd_par.write(fname)


def save_beam_as_h5(beam: Beam, fname: str):
    """
    Saves bmadx torch Beam as h5 file in openPMD-beamphysics
    ParticleGroup standard.

        Parameters
        ----------
        beam: bmax torch Beam to transform.

        fname: str
            file name 

        Returns
        -------
        None
    """
    pmd_par = bmadx_beam_to_openpmd(beam)
    pmd_par.write(fname)

def opal_data_to_bmadx_particle(
        opal_data_file: str, 
        p0c: float = None,
        mc2: float = M_ELECTRON
):
    """
    Transforms OPAL particle coordinates in a data file to
    Bmad-X Particle beam.

    Parameters
    ----------
    opal_data_file (str): OPAL data file with particle coordinates
    p0c (float): design momentum times c in eV as defined in Bmad coords
    mc2 (float): particle rest mass energy in eV
    
    Returns
    -------
    par (bmadx.Particle): Bmad-X Particle beam
    """

    data = np.genfromtxt(opal_data_file, skip_header=1)

    pc = mc2 * np.sqrt(data[:,1]**2 + data[:,3]**2 + data[:,5]**2)

    # if not provided, reference momentum is avg p
    if p0c is None:
        p0c = pc.mean()

    # initial transforms
    x = data[:,0]
    y = data[:,2]
    px = mc2 * data[:,1] / p0c 
    py = mc2 * data[:,3] / p0c
    pz = pc / p0c - 1.0

    # drift to z_avg (so that s value is the same)
    z0 = data[:,4].mean()
    dt = (z0 - data[:,4]) / data[:,5]
    x = x + data[:,1] * dt
    y = y + data[:,3] * dt

    # transform z coord
    beta = np.sqrt( 
        (pc / mc2)**2 / 
        ( 1 + (pc / mc2)**2 )
    )
    z = - beta * C_LIGHT * dt

    # return bmadx particle
    par = Particle(
        x = x,
        px = px,
        y = y,
        py = py,
        z = z,
        pz = pz,
        s = 0.0,
        p0c = p0c,
        mc2 = mc2
    )
    return par