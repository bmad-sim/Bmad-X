from bmadx.structures import Particle
from bmadx.constants import PI
from bmadx.low_level.offset_particle import make_offset_particle

#fringe_at = both_ends (default), no_end, entrance_end, exit_end 
#fringe_type = none (default), hard_edge_only, soft_edge_only, full

#--------------------
# TRACKING ROUTINES
#--------------------

def make_track_a_bend(lib):
    
    body, ent_linear, exit_linear, ent_hard, exit_hard,  ent_soft, exit_soft = make_track_a_sbend_parts(lib)
    offset_particle_set = make_offset_particle(lib, 'set')
    offset_particle_unset = make_offset_particle(lib, 'unset')
    
    def track_a_bend(p_in, bend):
        fringe_type = bend.FRINGE_TYPE.lower()
        fringe_at = bend.FRINGE_AT.lower()
        
        par = offset_particle_set(0.0, 0.0, bend.TILT, p_in)
        
        if fringe_type == "none" :
            p1 = body(par, bend)
            return offset_particle_unset(0.0, 0.0, bend.TILT, p1)
        
        elif fringe_type == "linear_edge":

            if fringe_at == "both_ends":
                p1 = ent_linear(par, bend)
                p2 = body(p1, bend)
                p3 = exit_linear(p2, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p3)
            elif fringe_at == "entrance_end":
                p1 = ent_linear(par, bend)
                p2 = body(p1, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p2)
            elif fringe_at == "exit_end":
                p1 = body(par, bend)
                p2 = exit_linear(p1, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p2)
            elif fringe_at=="no_end":
                p1 = body(par, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p1)
            else:
                raise ValueError(f"Unknown fringe_at setting {fringe_at}!!")
            
        elif fringe_type == "hard_edge_only":

            if fringe_at == "both_ends":
                p1 = ent_hard(par, bend)
                p2 = body(p1, bend)
                p3 = exit_hard(p2, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p3)
            elif fringe_at == "entrance_end":
                p1 = ent_hard(par, bend)
                p2 = body(p1, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p2)
            elif fringe_at == "exit_end":
                p1 = body(par, bend)
                p2 = exit_hard(p1, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p2)
            elif fringe_at=="no_end":
                p1 = body(par, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p1)
            else:
                raise ValueError(f"Unknown fringe_at setting {fringe_type}!!")
            
        elif fringe_type == "soft_edge_only": 
        
            if fringe_at == "both_ends":
                p1 = ent_soft(par, bend)
                p2 = body(p1, bend)
                p3 = exit_soft(p2, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p3)
            elif fringe_at == "entrance_end":
                p1 = ent_soft(par, bend)
                p2 = body(p1, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p2)
            elif fringe_at == "exit_end":
                p1 = body(par, bend)
                p2 = exit_soft(p1, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p2)
            elif fringe_at=="no_end":
                p1 = body(par, bend)
                return offset_particle_unset(0.0, 0.0, bend.TILT, p1)
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
    sinc = lib.sinc
    absolute = lib.abs
    arcsin = lib.arcsin
    arctan2 = lib.arctan2

    def sinc_bmad(x):
        return sinc(x/PI)
    
    def cosc(x):
        # FIX to make it branchless: (cos(x)-1)/x**2 = -1/2 [sinc(x/2)]**2
        return -sinc(x/PI/2)**2/2
        #if x == 0:
        #    return -0.5
        #else:
        #    return (cos(x) - 1) / x**2
        #eps = 2.220446049250313e-16
        #x = x + eps
        #return (cos(x) - 1) / x**2
    
    def track_body(p_in, bend):
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
        
        x2 = c1 * (temp < PI/2) + c2 * (temp >= PI/2)
        

        Lcu = x2 - L ** 2 * g * cosc(theta) - x * cos(theta)
        Lcv = -L * sinc_bmad(theta) - x * sin(theta)

        theta_p = 2 * (theta + phi1 - PI / 2 - arctan2(Lcv, Lcu))

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

    def track_entrance_linear(p_in, bend):
        g = bend.G
        dg = bend.DG
        e1 = bend.E1
        f_int = bend.F_INT
        h_gap = bend.H_GAP

        g_tot = g + dg
        hx = g_tot * tan(e1)
        hy = -g_tot*tan(
            e1 -
            2 * f_int * h_gap * g_tot * ( 1 + sin(e1)**2 ) / 
            cos(e1)
        )
        px_f = p_in.px + p_in.x * hx
        py_f = p_in.py + p_in.y * hy

        return Particle(
            p_in.x, 
            px_f, 
            p_in.y, 
            py_f, 
            p_in.z, 
            p_in.pz, 
            p_in.s, 
            p_in.p0c, 
            p_in.mc2)
    
    def track_exit_linear(p_in, bend):
        g = bend.G
        dg = bend.DG
        e2 = bend.E2
        f_int = bend.F_INT_X
        h_gap = bend.H_GAP_X

        g_tot = g + dg
        hx = g_tot * tan(e2)
        hy = -g_tot * tan(
            e2 -
            2 * f_int * h_gap * g_tot * ( 1 + sin(e2)**2 ) / 
            cos(e2)
        )
        px_f = p_in.px + p_in.x * hx
        py_f = p_in.py + p_in.y * hy

        return Particle(
            p_in.x, 
            px_f, 
            p_in.y, 
            py_f, 
            p_in.z, 
            p_in.pz, 
            p_in.s, 
            p_in.p0c, 
            p_in.mc2)

    def track_entrance_hard(p_in, bend):
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

        #Sigma_M1 = (
        #    (x**2 - y**2) * g_tot*tan1/2
        #    + y**2 * g_tot**2 * sec1**3 * (1 + sin1**2) * f_int*h_gap /2/(1 + pz)
        #    - x**3 * g_tot**2 * tan1**3 /12/(1 + pz)
        #    + x*y**2 * g_tot**2 * tan1 * sec1**2/4/(1 + pz)
        #    + (x**2 * px - x*y*py) * g_tot*tan1**2 /2/(1 + pz)
        #    + y**2 * px * g_tot * (1 + tan1**2) / 2 / (1 + pz)
        #)
                
        #Sigma_M1 = (
        #    (x**2 - y**2) * g_tot*tan1/2
        #    + y**2 * g_tot**2 * sec1**3 * (1 + sin1**2) * f_int*h_gap /2/(1 + pz)
        #    - x**3 * g_tot**2 * tan1**3 /12/(1 + pz)
        #    + x*y**2 * g_tot**2 * tan1 * sec1**2/4/(1 + pz)
        #    + (x**2 * px - 2*x*y*py) * g_tot*tan1**2 /2/(1 + pz)
        #    - y**2 * px * g_tot * (1 + tan1**2) / 2 / (1 + pz)
        #)

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

    
    def track_exit_hard(p_in, bend):
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
            + x*y**2 * g_tot**2 * tan2 * sec2**2 / 4 / (1 + pz)
            + (-x**2 * px + x*y*py) * g_tot * tan2**2 / 2 / (1 + pz)
            + y**2*px * g_tot * (1 + tan2**2) / 2 / (1 + pz)
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

    def track_entrance_soft(p_in, bend):
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

    def track_exit_soft(p_in, bend):
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
    
    return track_body, track_entrance_linear, track_exit_linear, track_entrance_hard, track_exit_hard, track_entrance_soft, track_exit_soft


