import sys

from bmadx.bmad.low_level.track_a_drift import make_track_a_drift
from bmadx.bmad.low_level.track_a_quadrupole import make_track_a_quadrupole
from bmadx.bmad.low_level.track_a_crab_cavity import make_track_a_crab_cavity
from bmadx.bmad.low_level.track_a_rf_cavity import make_track_a_rf_cavity

def track_a_lattice(p_in,lattice):
    """Tracks an incomming Particle p_in through lattice and returns a
    list of outgoing particles after each element.
    """
    lib = sys.modules[type(p_in.x).__module__]
    tracking_function_dict = {
        "Drift" : make_track_a_drift(lib),
        "Quadrupole" : make_track_a_quadrupole(lib),
        "CrabCavity" : make_track_a_crab_cavity(lib),
        "RFCavity" : make_track_a_rf_cavity(lib)
    }
    n = len(lattice)
    all_p = [None] * (n+1)
    all_p[0] = p_in
    
    for i in range(n):
        ele = lattice[i]
        track_f = tracking_function_dict[type(ele).__name__]
        all_p[i+1] = track_f(all_p[i],ele)
        
    return all_p