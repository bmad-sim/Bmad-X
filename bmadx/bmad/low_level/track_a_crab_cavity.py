from bmadx.bmad.modules import Particle, Drift
from bmadx.bmad.code.offset_particle import make_offset_particle
from bmadx.bmad.code.particle_rf_time import make_particle_rf_time
from bmadx.bmad.low_level.track_a_drift import make_track_a_drift
from bmadx.sim_utils.interfaces.constants import C_LIGHT, PI

def make_track_a_crab_cavity(lib):
    """Makes track_a_crab_cavity given the library lib."""
    sqrt = lib.sqrt
    sin = lib.sin
    cos = lib.cos
    track_this_drift = make_track_a_drift(lib)
    offset_particle_set = make_offset_particle(lib, 'set')
    offset_particle_unset = make_offset_particle(lib, 'unset')
    particle_rf_time = make_particle_rf_time(lib)
    
    def track_a_crab_cavity(p_in, cav):
        """Tracks an incomming Particle p_in through crab cavity and
        returns the ourgoing particle. 
        See Bmad manual section 4.9
        """
        s = p_in.s
        p0c = p_in.p0c
        mc2 = p_in.mc2
        
        l = cav.L
        
        x_off = cav.X_OFFSET
        y_off = cav.Y_OFFSET
        tilt = cav.TILT
        
        par = offset_particle_set(x_off, y_off, tilt, p_in)
        
        par = track_this_drift(par, Drift(l/2))
        x, px, y, py, z, pz = par.x, par.px, par.y, par.py, par.z, par.pz
        
        voltage = cav.VOLTAGE / p0c
        k_rf = 2 * PI * cav.RF_FREQUENCY / C_LIGHT
        phase = 2 * PI * (cav.PHI0 - (particle_rf_time(par)*cav.RF_FREQUENCY))
        
        px = px + voltage * sin(phase)
        
        beta = (1+pz) * p0c / sqrt(((1+pz)*p0c)**2 + mc2**2)
        beta_old = beta
        E_old =  (1+pz) * p0c / beta_old
        E_new = E_old + voltage * cos(phase) * k_rf * x * p0c
        pc = sqrt(E_new**2-mc2**2)
        beta = pc / E_new
        
        pz = (pc - p0c)/p0c        
        z = z * beta / beta_old
        
        par = track_this_drift(Particle(x, px, y, py, z, pz, s, p0c, mc2),
                               Drift(l/2))
        
        par = offset_particle_unset(x_off, y_off, tilt, par)
        
        return par
        
    return track_a_crab_cavity