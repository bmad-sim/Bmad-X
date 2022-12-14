from bmadx.bmad.modules import Particle, Drift
from bmadx.bmad.code.offset_particle import make_offset_particle
from bmadx.bmad.code.particle_rf_time import make_particle_rf_time
from .track_a_drift import make_track_a_drift
from .apply_energy_kick import make_apply_energy_kick
from bmadx.sim_utils.special_functions.sqrt_one import make_sqrt_one
from bmadx.sim_utils.interfaces.constants import C_LIGHT, PI

def make_track_a_rf_cavity(lib):
    """Makes track_a_crab_cavity given the library lib."""
    sqrt = lib.sqrt
    sin = lib.sin
    cos = lib.cos
    track_this_drift = make_track_a_drift(lib)
    offset_particle_set = make_offset_particle(lib, 'set')
    offset_particle_unset = make_offset_particle(lib, 'unset')
    particle_rf_time = make_particle_rf_time(lib)
    sqrt_one = make_sqrt_one(lib)
    apply_energy_kick = make_apply_energy_kick(lib)
    
    def track_a_rf_cavity(p_in, cav):
        """Tracks an incomming Particle p_in through rf cavity and
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
        
        voltage = cav.VOLTAGE
        k_rf = 2 * PI * cav.RF_FREQUENCY / C_LIGHT
        phase = 2 * PI * (cav.PHI0 - (particle_rf_time(par)*cav.RF_FREQUENCY))
        
        dE = voltage * sin(phase) / 2
        
        par = apply_energy_kick(dE, par)
        
        z_old = par.z
        
        par = track_this_drift(par, Drift(l))
        x, px, y, py, z, pz = par.x, par.px, par.y, par.py, par.z, par.pz
        beta_new = (1+pz) * p0c / sqrt(((1+pz)*p0c)**2 + mc2**2)
        phase = phase + 2 * PI * cav.RF_FREQUENCY * (z-z_old)/(C_LIGHT*beta_new)
        
        dE = voltage * sin(phase) / 2
        
        par = apply_energy_kick(dE, par)
        
        par = offset_particle_unset(x_off, y_off, tilt, par)
        
        return par
    
    return track_a_rf_cavity