def make_low_energy_z_correction(lib): 
    """Makes function given library lib"""
    sqrt = lib.sqrt
    
    def low_energy_z_correction(pz, p0c, mass, ds):
        """Corrects the change in z-coordinate due to speed < c_light.
        Input:
            p0c -- reference particle momentum in eV
            mass -- particle mass in eV
        Output: 
            dz -- dz=(ds-d_particle) + ds*(beta - beta_ref)/beta_ref
        """
        beta = (1+pz) * p0c / sqrt(((1+pz)*p0c)**2 + mass**2)
        beta0 = p0c / sqrt( p0c**2 + mass**2)
        e_tot = sqrt(p0c**2+mass**2)
        
        evaluation = mass * (beta0*pz)**2
        dz = (ds * pz * (1 - 3*(pz*beta0**2)/2+pz**2*beta0**2
                         * (2*beta0**2-(mass/e_tot)**2/2) )
              * (mass/e_tot)**2
              * (evaluation<3e-7*e_tot)
              + (ds*(beta-beta0)/beta0)
              * (evaluation>=3e-7*e_tot) )
        
        return dz
    
    return low_energy_z_correction