import sys
from bmadx import LIB_DICT

COORDS = ('x', 'px', 'y', 'py', 'z', 'pz')

def track_element(p_in, ele):
    """Tracks Particle 'p_in' though element 'ele'.
    Returns outgoing Particle.
    """
    lib = sys.modules[type(p_in.x).__module__]
    lib_ele = make_element(ele, lib)
    track_f = LIB_DICT[lib]['tracking_routine'][type(ele).__name__]
    return track_f(p_in, lib_ele)

def track_lattice(p_in, ele_list):
    """Tracks particle 'p_in' though list of elements 'ele_list'.
    Returns outgoing particle
    """
    p_out = p_in
    for ele in ele_list:
        p_out = track_element(p_out, ele)
    return p_out

def track_lattice_save_stats(p_in, ele_list, n_slices: int = 1):
    """Tracks particle 'p_in' though list of elements 'ele_list'.
    Returns dictionary with stats
    """
    sliced_lattice = stub_lattice(ele_list, n_slices)
    n = len(sliced_lattice) + 1
    stats = {}
    stats['s'] = [p_in.s] * n
    stats['p0c'] = [p_in.p0c] * n
    for coord in COORDS:
        array = getattr(p_in, coord)
        stats['mean_'+coord] = [array.mean()] * n
        stats['sigma_'+coord] = [array.std()] * n
        stats['min_'+coord] = [array.min()] * n
        stats['max_'+coord] = [array.max()] * n
        
    p_out = p_in
    
    for i, ele in enumerate(sliced_lattice):
        p_out = track_element(p_out, ele)
        stats['s'][i+1] = p_out.s
        stats['p0c'][i+1] = p_out.p0c
        for coord in COORDS:
            array = getattr(p_out, coord)
            stats['mean_'+coord][i+1] = array.mean()
            stats['sigma_'+coord][i+1] = array.std()
            stats['min_'+coord][i+1] = array.min()
            stats['max_'+coord][i+1] = array.max()

    stats['p_out'] = p_out
            
    return stats

def track_lattice_save_particles(p_in, ele_list, n_slices: int = 1):
    """Tracks particle 'p_in' though list of elements 'ele_list'.
    Returns list of Particles. all_p[0] is the incoming particle, 
    and all_p[i] is the particle after the 'i'th element slice.
    """
    sliced_lattice = stub_lattice(ele_list, n_slices)
    all_p = [None] * ( len(sliced_lattice) + 1 )
    all_p[0] = p_in
    lib = sys.modules[type(p_in.x).__module__]
    for i, ele in enumerate(sliced_lattice):
        all_p[i+1] = track_element(all_p[i], ele)
    return all_p

def make_element(ele, lib):
    """makes element named tuple with correct variable types
    given library 'lib'."""
    params = [*ele]
    for i, param in enumerate(params):
        if sys.modules[type(param).__module__] != lib:
            params[i] = LIB_DICT[lib]['construct_type'](param)
            
    return ele._make(params)

def stub_lattice(lattice, n):
    """Divides every element in the lattice into 'n' elements
    each and returns divided lattice. 
    **NOTE**: only use with Drifts and Quads as of now
    """
    stubbed_lattice = []
    
    for ele in lattice:
        stubbed_lattice.extend(stub_element(ele, n))
        
    return stubbed_lattice

def stub_element(ele, n):
    """Divides ele into 'n' equal length elements and returns
    a list of these short elements.
    **NOTE**: only use with Drifts and Quads with no fringe as of now
    """
    short_L = ele.L / n
    short_ele = ele._replace(L=short_L)
    lattice = [short_ele] * n
    
    return lattice