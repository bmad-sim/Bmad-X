from collections import namedtuple
import sys
import math
import torch
tkwargs = {
    "dtype" : torch.double
}

from bmadx.track import Particle

#--------------------
# Named tuples for elements and particles
#--------------------

# Particles
# Phase space coordinates as def. in Bmad manual section 15.4.2


# Elements

#Quadrupole = namedtuple('Quadrupole', 
#                        'L K1 NUM_STEPS X_OFFSET Y_OFFSET, TILT',
#                        defaults=(None, None, 1, 0, 0, 0))

Sbend = namedtuple('Sbend', 
                        'L P0C G DG E1 E2 F_INT H_GAP F_INT_X H_GAP_X FRINGE_AT FRINGE_TYPE',
                        defaults=(None, None, 0, 0, 0, 0, 0, 0, 0, 0, "both_ends", "none"))

# from bmad
#fringe_at = both_ends (default), no_end, entrance_end, exit_end 
#fringe_type = none (default), hard_edge_only, soft_edge_only, full

#--------------------
# TRACKING ROUTINES
#--------------------

def make_track_a_bend(lib):
    
    body, ent_hard, exit_hard,  ent_soft, exit_soft = make_track_a_sbend_parts(lib)
    
    def track_a_bend(p_in, bend: Sbend):
        
        fringe_type = bend.FRINGE_TYPE.lower()
        fringe_at = bend.FRINGE_AT.lower()
        
        
        if fringe_type == "none" :
            return body(p_in, bend)
        elif fringe_type == "hard_edge_only":

            if fringe_at == "both_ends":
                p1 = ent_hard(p_in, bend)
                p2 = body(p1, bend)
                return exit_hard(p2, bend)
            elif fringe_at == "entrance_end":
                p1 = ent_hard(p_in, bend)
                return body(p1, bend)
            elif fringe_at == "exit_end":
                p1 = body(p_in, bend)
                return exit_hard(p1, bend)
            elif fringe_at=="no_end":
                return body(p1, bend)
            else:
                raise ValueError(f"Unknown fringe_at setting {fringe_at}!!")
            
        elif fringe_type == "soft_edge_only":   
            
            if fringe_at == "both_ends":
                p1 = ent_soft(p_in, bend)
                p2 = body(p1, bend)
                return exit_soft(p2, bend)
            elif fringe_at == "entrance_end":
                p1 = ent_soft(p_in, bend)
                return body(p1, bend)
            elif fringe_at == "exit_end":
                p1 = body(p_in, bend)
                return exit_soft(p1, bend)
            elif fringe_at=="no_end":
                return body(p1, bend)
            else:
                raise ValueError(f"Unknown fringe_at setting {fringe_at}!!")
                
        elif fringe_type == "full":
            raise NotImplementedError("FULL FRINGE to be implemented...")
        else:
            raise ValueError(f"Unknown fringe_at setting {fringe_type}!!")
    return track_a_bend


def make_track_a_sbend_parts(lib):
    """Makes track_a_quadrupole given the library lib."""
    sqrt = lib.sqrt
    sin = lib.sin
    cos = lib.cos
    tan = lib.tan
    sinh = lib.sinh
    cosh = lib.cosh
    sinc = lib.sinc
    pi = math.pi
    absolute = lib.abs
    arcsin = lib.arcsin
    arctan2 = lib.arctan2

    def sinc_bmad(x):
        return sinc(x /pi)
    
    def cosc(x):
        if x == 0:
            return -0.5
        else:
            return (cos(x) - 1) / x**2
    
    def track_body(p_in, bend: Sbend):
        """Tracks the incoming Particle p_in though a sbend (k1=0) element
        and returns the outgoing particle. 
        See Bmad manual section 24.9 
        """
       
        # WILLIAM's CODE HERE
        L = bend.L
        g = bend.G
        dg = bend.DG
        p0c = bend.P0C
        
        p0c_particle = p_in.p0c 
        
        x = p_in.x
        px = p_in.px * p0c_particle / p0c
        y = p_in.y
        py = p_in.py * p0c_particle / p0c
        z = p_in.z
        pz = (p_in.pz + 1) * p0c_particle / p0c - 1
        s = p_in.s
        mc2 = p_in.mc2
        
        # Calculate the final 6 corrds

        px_norm = sqrt((1 + pz) ** 2 - py ** 2)  # For simplicity

        phi1 = arcsin(px / px_norm)

        # g = theta / L
        theta = g*L
        g_tot = g + dg
        gp = g_tot / px_norm

        alpha = (2 * (1 + g * x) * sin(theta + phi1) * L * sinc_bmad(theta)
            - gp * ((1 + g * x) * L * sinc_bmad(theta)) ** 2)

        x2_t1 = x * cos(theta) + L ** 2 * g * cosc(theta)
        x2_t2 = sqrt((cos(theta + phi1) ** 2) + gp * alpha)
        x2_t3 = cos(theta + phi1)

        #x2 = where(absolute(theta + phi1) < pi / 2,
        #    (x2_t1 + alpha / (x2_t2 + x2_t3)),
        #    (x2_t1 + (x2_t2 - x2_t3) / gp))
        
        temp = absolute(theta + phi1)
        c1 = (x2_t1 + alpha / (x2_t2 + x2_t3))
        c2 = (x2_t1 + (x2_t2 - x2_t3) / gp)
        
        x2 = c1 * (temp < pi/2) + c2 * (temp >= pi/2)
        

        Lcu = x2 - L ** 2 * g * cosc(theta) - x * cos(theta)
        Lcv = -L * sinc_bmad(theta) - x * sin(theta)

        theta_p = 2 * (theta + phi1 - pi / 2 - arctan2(Lcv, Lcu))

        Lc = sqrt(Lcu ** 2 + Lcv ** 2)
        Lp = Lc / sinc_bmad(theta_p / 2)

        P = p0c * (1 + pz)  # in eV
        E = sqrt(P**2 + mc2**2)  # in eV
        E0 = sqrt(p0c**2 + mc2**2)  # in eV
        beta = P / E
        beta0 = p0c / E0

        xf = x2
        pxf = px_norm * sin(theta + phi1 - theta_p)
        yf = y + py * Lp / px_norm
        pyf = py
        zf = z + (beta * L / beta0) - ((1 + pz) * Lp / px_norm)
        pzf = pz

        #return np.array([xf, pxf, yf, pyf, zf, pzf])        
        
        
        s = s+L
        
        return Particle(xf, pxf, yf, pyf, zf, pzf, s, p0c, mc2)

    def track_entrance_hard(p_in, bend: Sbend):
        L = bend.L
        g = bend.G
        dg = bend.DG
        p0c = bend.P0C
        e1 = bend.E1
        f_int = bend.F_INT
        h_gap = bend.H_GAP
    
        p0c_particle = p_in.p0c 
        
        x = p_in.x
        px = p_in.px * p0c_particle / p0c
        y = p_in.y
        py = p_in.py * p0c_particle / p0c
        z = p_in.z
        pz = (p_in.pz + 1) * p0c_particle / p0c - 1
        s = p_in.s
        mc2 = p_in.mc2
    
        theta = g*L
        g_tot = g + dg

        sin1 = sin(e1)
        tan1 = tan(e1)
        sec1 = 1 / cos(e1)

        Sigma_M1 = (
            (x**2 - y**2) * g_tot*tan1/2
            + y**2 * g_tot**2 * sec1**3 * (1 + sin1**2) * f_int*h_gap /2/(1 + pz)
            - x**3 * g_tot**2 * tan1**3 /12/(1 + pz)
            + x*y**2 * g_tot**2 * tan1 * sec1**2/4/(1 + pz)
            + (x**2 * px - 2*x*y*py) * g_tot*tan1**2 /2/(1 + pz)
            - y**2 * px * g_tot * (1 + tan1**2) / 2 / (1 + pz)
        )

        xf = x + g_tot / 2 / (1 + pz) * (-x**2 * tan1**2 + y**2 * sec1**2)

        pxf = ( px
            + x * g_tot * tan1
            + y**2 * g_tot**2 * (tan1 + 2*tan1**3) /2/(1 + pz)
            + (x*px - y*py) * g_tot * tan1**2 /(1 + pz)
        )

        yf = y + x * y * g_tot * tan1 ** 2 / (1 + pz)

        pyf = ( py
            + y * ( -g_tot * tan1 + g_tot**2 * (1 + sin1**2) * sec1**3 * f_int * h_gap / (1 + pz))
            - x * py * g_tot * tan1**2 / (1 + pz)
            - y * px * g_tot * (1 + tan1**2) / (1 + pz)
        )

        zf = z + (Sigma_M1 - (x**2 - y**2) * g_tot * tan1/2) / (1 + pz)

        pzf = pz
        
        
        return Particle(xf, pxf, yf, pyf, zf, pzf, s, p0c, mc2)

    
    def track_exit_hard(p_in, bend: Sbend):
        L = bend.L
        g = bend.G
        dg = bend.DG
        p0c = bend.P0C
        e2 = bend.E2
        f_int = bend.F_INT_X
        h_gap = bend.H_GAP_X
        
        p0c_particle = p_in.p0c 
        
        x = p_in.x
        px = p_in.px * p0c_particle / p0c
        y = p_in.y
        py = p_in.py * p0c_particle / p0c
        z = p_in.z
        pz = (p_in.pz + 1) * p0c_particle / p0c - 1
        s = p_in.s
        mc2 = p_in.mc2
    
        theta = g*L
        g_tot = g + dg

        sin2 = sin(e2)
        tan2 = tan(e2)
        sec2 = 1 / cos(e2)
        
        
        Sigma_M2 = (
            (x**2 - y**2) * g_tot * tan2 / 2
            + y**2 * g_tot**2 * sec2**3 * (1 + sin2**2) * f_int * h_gap / 2 / (1 + pz)
            - x**3 * g_tot**2 * tan2**3 / 12 / (1 + pz)
            + x * y**2 * g_tot**2 * tan2 * sec2**2 / 4 / (1 + pz)
            + (-x**2 * px + 2 * x * y * py) * g_tot * tan2**2 / 2 / (1 + pz)
            + y**2 * px * g_tot * (1 + tan2**2) / 2 / (1 + pz)
        )

        xf = x + g_tot / 2 / (1 + pz) * (x**2 * tan2**2 - y ** 2 * sec2**2)

        pxf = (
            px + x * g_tot * tan2
            - (x**2 + y**2) * g_tot**2 * tan2**3 / 2 / (1 + pz)
            + (-x * px + y * py) * g_tot * tan2**2 / (1 + pz)
        )

        yf = y - x * y * g_tot * tan2**2 / (1 + pz)

        pyf = (
            py + y * ( -g_tot * tan2 + g_tot**2 * (1 + sin2**2) * sec2**3 * f_int * h_gap / (1 + pz) )
            + x * y * g_tot**2 * sec2**2 * tan2 / (1 + pz)
            + x * py * g_tot * tan2**2 / (1 + pz)
            + y * px * g_tot * (1 + tan2**2) / (1 + pz)
        )

        zf = z + (Sigma_M2 - (x**2 - y**2) * g_tot * tan2 / 2) / (1 + pz)

        pzf = pz

        return Particle(xf, pxf, yf, pyf, zf, pzf, s, p0c, mc2)

    def track_entrance_soft(p_in, bend: Sbend):
        L = bend.L
        g = bend.G
        dg = bend.DG
        p0c = bend.P0C
        e1 = bend.E1
        f_int = bend.F_INT
        h_gap = bend.H_GAP
    
        p0c_particle = p_in.p0c 
        
        x = p_in.x
        px = p_in.px * p0c_particle / p0c
        y = p_in.y
        py = p_in.py * p0c_particle / p0c
        z = p_in.z
        pz = (p_in.pz + 1) * p0c_particle / p0c - 1
        s = p_in.s
        mc2 = p_in.mc2

        g_tot = g + dg
        
        FH1 = f_int*h_gap
        c1 = 6*g_tot * FH1**2/(1+pz)
        c2 = 2*g_tot**2 * FH1/(1+pz)
        c3 = g_tot**2 /18 /FH1/(1+pz)

        xf = x + c1*pz

        pxf = px

        yf = y

        pyf = py + c2*y - c3*y**3

        zf = z + (c1*px + c2*y**2/2 - c3*y**4/4) / (1 + pz)

        pzf = pz
        
        return Particle(xf, pxf, yf, pyf, zf, pzf, s, p0c, mc2)    

    def track_exit_soft(p_in, bend: Sbend):
        L = bend.L
        g = bend.G
        dg = bend.DG
        p0c = bend.P0C
        e1 = bend.E1
        f_int = bend.F_INT_X
        h_gap = bend.H_GAP_X
    
        p0c_particle = p_in.p0c 
        
        x = p_in.x
        px = p_in.px * p0c_particle / p0c
        y = p_in.y
        py = p_in.py * p0c_particle / p0c
        z = p_in.z
        pz = (p_in.pz + 1) * p0c_particle / p0c - 1
        s = p_in.s
        mc2 = p_in.mc2

        g_tot = -g - dg # sign reversed for soft exit fringe
        
        FH1 = f_int*h_gap
        c1 = 6*g_tot * FH1**2/(1+pz)
        c2 = 2*g_tot**2 * FH1/(1+pz)
        c3 = g_tot**2 /18 /FH1/(1+pz)

        xf = x + c1*pz

        pxf = px

        yf = y

        pyf = py + c2*y - c3*y**3

        zf = z + (c1*px + c2*y**2/2 - c3*y**4/4) / (1 + pz)

        pzf = pz
        
        return Particle(xf, pxf, yf, pyf, zf, pzf, s, p0c, mc2) 
    
    return track_body, track_entrance_hard, track_exit_hard, track_entrance_soft, track_exit_soft






def make_track_a_drift(lib):
    """Makes track_a_drift given the library lib."""
    sqrt = lib.sqrt
    
    def track_a_drift(p_in, drift):
        """Tracks the incoming Particle p_in though drift element
        and returns the outgoing particle. 
        See Bmad manual section 24.9 
        """
        L = drift.L
        
        s = p_in.s
        p0c = p_in.p0c
        mc2 = p_in.mc2
        
        x = p_in.x
        px = p_in.px
        y = p_in.y
        py = p_in.py
        z = p_in.z
        pz = p_in.pz
        
        P = 1 + pz            # Particle's total momentum over p0
        Px = px / P           # Particle's 'x' momentum over p0
        Py = py / P           # Particle's 'y' momentum over p0
        Pxy2 = Px**2 + Py**2  # Particle's transverse mometum^2 over p0^2
        Pl = sqrt(1-Pxy2)     # Particle's longitudinal momentum over p0
        
        x = x + L * Px / Pl
        y = y + L * Py / Pl
        
        beta = P * p0c / sqrt( (P*p0c)**2 + mc2**2)
        beta_ref = p0c / sqrt( p0c**2 + mc2**2)
        z = z + L * ( beta/beta_ref - 1.0/Pl )
        s = s + L
        
        return Particle(x, px, y, py, z, pz, s, p0c, mc2)
    
    return track_a_drift

def make_track_a_quadrupole(lib):
    """Makes track_a_quadrupole given the library lib."""
    sqrt = lib.sqrt
    sin = lib.sin
    cos = lib.cos
    sinh = lib.sinh
    cosh = lib.cosh
    sinc = lib.sinc
    pi = math.pi
    absolute = lib.abs
    
    def quad_mat2_calc(k1, length, rel_p):
        """Returns 2x2 transfer matrix elements aij and the coefficients
        to calculate the change in z position.
        Input: 
            k1_ref -- Quad strength: k1 > 0 ==> defocus
            length -- Quad length
            rel_p -- Relative momentum P/P0
        Output:
            a11, a12, a21, a22 -- transfer matrix elements
            c1, c2, c3 -- second order derivatives of z such that 
                           z = c1 * x_0^2 + c2 * x_0 * px_0 + c3* px_0^2
        **NOTE**: accumulated error due to machine epsilon. REVISIT
        """ 
        eps = 2.220446049250313e-16  # machine epsilon to double precission
        
        sqrt_k = sqrt(absolute(k1)+eps)
        sk_l = sqrt_k * length
        
        cx = cos(sk_l) * (k1<=0) + cosh(sk_l) * (k1>0) 
        sx = (sin(sk_l)/(sqrt_k))*(k1<=0) + (sinh(sk_l)/(sqrt_k))*(k1>0)
          
        a11 = cx
        a12 = sx / rel_p
        a21 = k1 * sx * rel_p
        a22 = cx
            
        c1 = k1 * (-cx * sx + length) / 4
        c2 = -k1 * sx**2 / (2 * rel_p)
        c3 = -(cx * sx + length) / (4 * rel_p**2)

        return a11, a12, a21, a22, c1, c2, c3
    
    def low_energy_z_correction(pz, p0c, mass, ds):
        """Corrects the change in z-coordinate due to speed < c_light.
        Input:
            p0c -- reference particle momentum in eV
            mass -- particle mass in eV
        Output: 
            dz -- dz = (ds - d_particle) + ds*(beta - beta_ref)/beta_ref
        """
        beta = (1+pz) * p0c / sqrt(((1+pz)*p0c)**2 + mass**2)
        beta0 = p0c / sqrt( p0c**2 + mass**2)
        e_tot = sqrt(p0c**2+mass**2)
        
        evaluation = mass * (beta0*pz)**2
        dz = ( (ds*pz*(1-3*(pz*beta0**2)/2+pz**2*beta0**2*(2*beta0**2-(mass/e_tot)**2/2))*(mass/e_tot)**2)
              * (evaluation<3e-7*e_tot)
              + (ds*(beta-beta0)/beta0)
              * (evaluation>=3e-7*e_tot) )
        
        return dz
    
    def offset_particle_entrance(length, x_offset, y_offset, tilt, 
                                 x_lab, y_lab, px_lab, py_lab):
        """transform from the laboratory coordinates to the
        entrance element coordinates.
        See Bmad sections 5.6.1, 15.3.1 and 24.2
        **NOTE**: transverse only.
        """
        s = sin(tilt)
        c = cos(tilt)
        x_ele_int = x_lab - x_offset
        y_ele_int = y_lab - y_offset
        x_ele = x_ele_int*c + y_ele_int*s
        y_ele = -x_ele_int*s + y_ele_int*c
        px_ele = px_lab*c + py_lab*s
        py_ele = -px_lab*s + py_lab*c
        return x_ele, y_ele, px_ele, py_ele
    
    def offset_particle_exit(length, x_offset, y_offset, tilt, 
                             x_ele, y_ele, px_ele, py_ele):
        """Transforms from the exit element reference frame to the
        laboratory reference frame.
        See Bmad sections 5.6.1, 15.3.1 and 24.2
        **NOTE**: transverse only as of now.
        """
        s = sin(tilt)
        c = cos(tilt)
        x_lab_int = x_ele*c - y_ele*s
        y_lab_int = x_ele*s + y_ele*c
        x_lab = x_lab_int + x_offset
        y_lab = y_lab_int + y_offset
        px_lab = px_ele*c - py_ele*s
        py_lab = px_ele*s + py_ele*c 
        return x_lab, y_lab, px_lab, py_lab
    
    def track_a_quadrupole(p_in, quad):
        """Tracks the incoming Particle p_in though quad element and 
        returns the outgoing particle.
        See Bmad manual section 24.15
        """
        l = quad.L
        k1 = quad.K1
        n_step = quad.NUM_STEPS  # number of divisions
        step_len = l/n_step  # length of division
        
        x_off = quad.X_OFFSET
        y_off = quad.Y_OFFSET
        tilt = quad.TILT
        
        b1 = k1*l
        
        s = p_in.s
        p0c = p_in.p0c
        mc2 = p_in.mc2
        
        x = p_in.x
        px = p_in.px
        y = p_in.y
        py = p_in.py
        z = p_in.z
        pz = p_in.pz
        
        # --- TRACKING --- :
        
        x, y, px, py = offset_particle_entrance(l, x_off, y_off, tilt, x, y, px, py)
        
        for i in range(n_step):
            rel_p = 1 + pz  # Particle's relative momentum (P/P0)
            k1 = b1/(l*rel_p)
            
            tx11, tx12, tx21, tx22, dz_x1, dz_x2, dz_x3 = quad_mat2_calc(-k1, step_len, rel_p)
            ty11, ty12, ty21, ty22, dz_y1, dz_y2, dz_y3 = quad_mat2_calc( k1, step_len, rel_p)
            
            z = ( z +
                dz_x1 * x**2 + dz_x2 * x * px + dz_x3 * px**2 +
                dz_y1 * y**2 + dz_y2 * y * py + dz_y3 * py**2 )
            x_next = tx11 * x + tx12 * px
            px_next = tx21 * x + tx22 * px
            y_next = ty11 * y + ty12 * py
            py_next = ty21 * y + ty22 * py
            x = x_next
            px = px_next
            y = y_next
            py = py_next
            
            z = z + low_energy_z_correction(pz, p0c, mc2, step_len)
        
        s = s + l
        
        x, y, px, py = offset_particle_exit(l, x_off, y_off, tilt, x, y, px, py)
        
        return Particle(x, px, y, py, z, pz, s, p0c, mc2)
      
    return track_a_quadrupole

def track_a_lattice(p_in,lattice):
    """Tracks an incomming Particle p_in through lattice and returns a 
    list of outgoing particles after each element.
    """
    lib = sys.modules[type(p_in.x).__module__]
    tracking_function_dict = {
        "Drift" : make_track_a_drift(lib),
        "Quadrupole" : make_track_a_quadrupole(lib)
    }
    n = len(lattice)
    all_p = [None] * (n+1)
    all_p[0] = p_in
    
    for i in range(n):
        ele = lattice[i]
        track_f = tracking_function_dict[type(ele).__name__]
        all_p[i+1] = track_f(all_p[i],ele)
        
    return all_p


def stub_element(ele, n):
    """Divides ele into 'n' equal length elements and returns
    a list of these short elements.
    """
    short_L = ele.L / n
    short_ele = ele._replace(L=short_L)
    lattice = [short_ele] * n
    
    return lattice

def stub_lattice(lattice, n):
    """Divides every element in the lattice into 'n' elements
    each and returns divided lattice. 
    """
    stubbed_lattice = []
    
    for ele in lattice:
        stubbed_lattice.extend(stub_element(ele, n))
        
    return stubbed_lattice