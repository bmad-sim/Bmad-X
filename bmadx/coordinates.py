from bmadx.constants import C_LIGHT, M_ELECTRON, E_CHARGE
import numpy as np
import torch
import sys

from pmd_beamphysics import ParticleGroup


def openpmd_to_bmadx(pmd_par, p0c):
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
    z = pmd_par.beta * C_LIGHT * pmd_par.t
    pz = pmd_par.p / p0c - 1.0

    bmad_coords = (x, px, y, py, z, pz)

    return bmad_coords

def bmadx_to_openpmd(par):
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
    if lib==np:
        x = par.x
        px = par.px
        y = par.y
        py = par.py
        z = par.z
        pz = par.pz
    elif lib==torch:
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
    dat['z'] = np.zeros_like(z)
    dat['pz'] = par.p0c * ( (pz + 1.0)**2 - px**2 - py**2 )**0.5
    dat['t'] = (z 
                / ( ((pz+1)*par.p0c/par.mc2)**2 
                   / (1 + ((pz+1)*par.p0c/par.mc2)**2) 
                   )**0.5
                / C_LIGHT
                )
    dat['status'] = np.ones_like(x, dtype=int)
    dat['weight'] = - np.ones_like(x, dtype=int) * E_CHARGE

    if np.isclose(par.mc2,M_ELECTRON):
        dat['species'] = 'electron'
    else:
        raise ValueError('only electrons are supported as of now')


    return ParticleGroup(data=dat)

def save_as_h5(par, fname: str):
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
    pmd_par = bmadx_to_openpmd(par)
    pmd_par.write(fname)