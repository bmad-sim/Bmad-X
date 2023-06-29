from bmadx.constants import C_LIGHT

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
