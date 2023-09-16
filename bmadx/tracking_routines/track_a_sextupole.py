from bmadx.bmad.modules import Particle
from bmadx.bmad.code.offset_particle import make_offset_particle
from bmadx.bmad.code.low_energy_z_correction import make_low_energy_z_correction

def make_track_a_sextupole(lib):
    """Makes track_a_quadrupole given the library lib."""
    offset_particle_set = make_offset_particle(lib, 'set')
    offset_particle_unset = make_offset_particle(lib, 'unset')
    low_energy_z_correction = make_low_energy_z_correction(lib)
    
    def track_a_sextupole(p_in, sextupole):
        """Tracks the incoming Particle p_in though pure sextupole element and
        returns the outgoing particle.
        See Bmad manual section 24.15
        """
        l = sextupole.L
        k2 = sextupole.K2

        n_step = sextupole.NUM_STEPS  # number of divisions
        step_len = l / n_step  # length of division
        
        x_off = sextupole.X_OFFSET
        y_off = sextupole.Y_OFFSET
        tilt = sextupole.TILT
        
        b2 = k2 * l
        
        s = p_in.s
        p0c = p_in.p0c
        mc2 = p_in.mc2
        
        # --- TRACKING --- :
        
        par = offset_particle_set(x_off, y_off, tilt, p_in)
        x, px, y, py, z, pz = par.x, par.px, par.y, par.py, par.z, par.pz
        
        for i in range(n_step):
            rel_p = 1 + pz  # Particle's relative momentum (P/P0)
            k2 = b2/(l*rel_p)

            x_next = x + step_len * px
            y_next = y + step_len * py

            px_next = px + 0.5 * k2 * step_len * (y**2 - x**2)
            py_next = py + k2 * step_len * x * y
            
            x, px, y, py = x_next, px_next, y_next, py_next
            
            z = z + low_energy_z_correction(pz, p0c, mc2, step_len)
        
        s = s + l
        
        par = offset_particle_unset(x_off, y_off, tilt,
                                    Particle(x, px, y, py, z, pz, s, p0c, mc2))
        
        return par
      
    return track_a_sextupole