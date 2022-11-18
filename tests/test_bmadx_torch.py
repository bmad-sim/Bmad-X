import pytest
from pytao import Tao
import torch
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
tkwargs = {
    "dtype" : torch.double
}
import warnings

from bmadx import *

def torch_quadrupole(L: torch.Tensor, K1: torch.Tensor, NUM_STEPS=1,
                     X_OFFSET: torch.Tensor=torch.tensor(0.0,**tkwargs),
                     Y_OFFSET: torch.Tensor=torch.tensor(0.0,**tkwargs),
                     TILT: torch.Tensor=torch.tensor(0.0,**tkwargs)):
    return Quadrupole(L=L, K1=K1, NUM_STEPS=NUM_STEPS,
                      X_OFFSET=X_OFFSET, Y_OFFSET=Y_OFFSET,
                      TILT=TILT)

def torch_crab_cavity(L: torch.Tensor, VOLTAGE: torch.Tensor,
                      PHI0: torch.Tensor, RF_FREQUENCY: torch.Tensor,
                      X_OFFSET: torch.Tensor=torch.tensor(0.0,**tkwargs),
                      Y_OFFSET: torch.Tensor=torch.tensor(0.0,**tkwargs),
                      TILT: torch.Tensor=torch.tensor(0.0,**tkwargs)):
    return CrabCavity(L=L, VOLTAGE=VOLTAGE, PHI0=PHI0,
                      RF_FREQUENCY=RF_FREQUENCY, X_OFFSET=X_OFFSET,
                      Y_OFFSET=Y_OFFSET, TILT=TILT)

def torch_rf_cavity(L: torch.Tensor, VOLTAGE: torch.Tensor,
                    PHI0: torch.Tensor, RF_FREQUENCY: torch.Tensor,
                    X_OFFSET: torch.Tensor=torch.tensor(0.0,**tkwargs),
                    Y_OFFSET: torch.Tensor=torch.tensor(0.0,**tkwargs),
                    TILT: torch.Tensor=torch.tensor(0.0,**tkwargs)):
    return RFCavity(L=L, VOLTAGE=VOLTAGE, PHI0=PHI0,
                    RF_FREQUENCY=RF_FREQUENCY, X_OFFSET=X_OFFSET,
                    Y_OFFSET=Y_OFFSET, TILT=TILT)

def set_tao(tao, pvec):
    tao.cmd('set particle_start x='+str(pvec[0]))
    tao.cmd('set particle_start px='+str(pvec[1]))
    tao.cmd('set particle_start y='+str(pvec[2]))
    tao.cmd('set particle_start py='+str(pvec[3]))
    tao.cmd('set particle_start z='+str(pvec[4]))
    tao.cmd('set particle_start pz='+str(pvec[5]))
    
track_a_drift_torch = make_track_a_drift(torch)
track_a_quadrupole_torch = make_track_a_quadrupole(torch)
track_a_crab_cavity_torch = make_track_a_crab_cavity(torch)
track_a_rf_cavity_torch = make_track_a_rf_cavity(torch)

class TestBmadxTorch:
    
    # Incoming particle
    s = 0.0 #initial s
    p0c = 4.0E+07 #Reference particle momentum in eV
    mc2 = 1*M_ELECTRON # electron mass in eV
    ts = torch.tensor(s, **tkwargs)
    tp0c = torch.tensor(p0c, **tkwargs)
    tmc2 = torch.tensor(mc2, **tkwargs)
    pvec1 = [2e-3,3e-3,-3e-3,-1e-3,2e-3,-2e-3] 
    tvec1 = torch.tensor(pvec1, requires_grad=True, **tkwargs)
    p_in = Particle(*tvec1, ts, tp0c, tmc2)
    
    def test_drift(self):
        # test one particle tracking
        L_d=1.0 # Drift length in m
        d1 = Drift(L=torch.tensor(L_d, **tkwargs))
        p_out = track_a_drift_torch(self.p_in, d1)
        x_py = torch.hstack([p_out.x,p_out.px,p_out.y,
                             p_out.py,p_out.z,p_out.pz]).detach()
        
        tao = Tao('-lat tests/bmad_lattices/test_drift.bmad -noplot')
        set_tao(tao, self.pvec1)
        orbit_out=tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_drift = lambda x: track_a_drift_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), d1)[:6]
        
        J = jacobian(f_drift, self.tvec1)
        mat_py = torch.vstack(J)
        drift_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(drift_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
        
    def test_quadrupole(self):
        # no offset nor tilt
        # test one particle tracking
        L_q = 0.1 #Length in m
        K1 = 10 #Quad focusing strength. Positive is focusing in x
        q1 = torch_quadrupole(L=torch.tensor(L_q, **tkwargs),
                             K1=torch.tensor(K1, **tkwargs))
        p_out = track_a_quadrupole_torch(self.p_in, q1)
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao('-lat tests/bmad_lattices/test_quad.bmad -noplot')
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_quadrupole = lambda x: track_a_quadrupole_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), q1)[:6]
        
        J = jacobian(f_quadrupole, self.tvec1)
        mat_py = torch.vstack(J)
        quad_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(quad_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
    
    def test_quadrupole_offset(self):
        # x-y offset, no tilt
        # test one particle tracking
        L_q = 0.1 # Length in m
        K1 = 10 # Quad focusing strength. Positive is focusing in x
        x_off = 1e-3
        y_off = -2e-3
        q1 = torch_quadrupole(L=torch.tensor(L_q, **tkwargs),
                                K1=torch.tensor(K1, **tkwargs),
                                X_OFFSET=torch.tensor(x_off, **tkwargs),
                                Y_OFFSET=torch.tensor(y_off, **tkwargs) )
        p_out = track_a_quadrupole_torch(self.p_in, q1)
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao('-lat tests/bmad_lattices/test_quad_offset.bmad -noplot')
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_quadrupole = lambda x: track_a_quadrupole_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), q1)[:6]
        
        J = jacobian(f_quadrupole, self.tvec1)
        mat_py = torch.vstack(J)
        quad_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(quad_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
        
    def test_quadrupole_tilt(self):
        # tilt, no offset
        # test one particle tracking
        L_q = 0.1 # Length in m
        K1 = 10 # Quad focusing strength. Positive is focusing in x
        tilt = 0.3 # transverse rotation in radians
        q1 = torch_quadrupole(L=torch.tensor(L_q, **tkwargs),
                                K1=torch.tensor(K1, **tkwargs),
                                TILT=torch.tensor(tilt, **tkwargs))
        p_out = track_a_quadrupole_torch(self.p_in, q1)
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao('-lat tests/bmad_lattices/test_quad_tilt.bmad -noplot')
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_quadrupole = lambda x: track_a_quadrupole_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), q1)[:6]
        
        J = jacobian(f_quadrupole, self.tvec1)
        mat_py = torch.vstack(J)
        quad_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(quad_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
        
    def test_crab_cavity(self):
        # no offset nor tilt
        # test one particle tracking
        L = 0.2 #Length in m
        voltage = 1.0e4 #Quad focusing strength. Positive is focusing in x
        phi0 = 0.5
        rf = 1.0e9
        cav = torch_crab_cavity(L=torch.tensor(L, **tkwargs),
                                VOLTAGE=torch.tensor(voltage, **tkwargs),
                                PHI0=torch.tensor(phi0, **tkwargs),
                                RF_FREQUENCY=torch.tensor(rf, **tkwargs))
        
        p_out = track_a_crab_cavity_torch(self.p_in, cav)
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao('-lat tests/bmad_lattices/test_crab_cavity.bmad -noplot')
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_cav = lambda x: track_a_crab_cavity_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), cav)[:6]
        
        J = jacobian(f_cav, self.tvec1)
        mat_py = torch.vstack(J)
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
    
    def test_crab_cavity_offset(self):
        # x-y offset, no tilt
        # test one particle tracking
        L = 0.2
        voltage = 1.0e3
        phi0 = 0.5
        rf = 1.0e9
        x_off = 1e-3
        y_off = -2e-3
        cav_off = torch_crab_cavity(L=torch.tensor(L, **tkwargs),
                                    VOLTAGE=torch.tensor(voltage, **tkwargs),
                                    PHI0=torch.tensor(phi0, **tkwargs),
                                    RF_FREQUENCY=torch.tensor(rf, **tkwargs),
                                    X_OFFSET=torch.tensor(x_off, **tkwargs),
                                    Y_OFFSET=torch.tensor(y_off, **tkwargs))
        
        p_out = track_a_crab_cavity_torch(
            Particle(*self.tvec1, self.ts, self.tp0c, self.tmc2),
            cav_off)
        
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_crab_cavity_offset.bmad -noplot')
        
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_cav = lambda x: track_a_crab_cavity_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), cav_off)[:6]
        
        J = jacobian(f_cav, self.tvec1)
        mat_py = torch.vstack(J)
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
        
    def test_crab_cavity_tilt(self):
        # tilt, no offset
        # test one particle tracking
        L = 0.2
        voltage = 1.0e3
        phi0 = 0.5
        rf = 1.0e9
        tilt = 0.3
        cav_tilt = torch_crab_cavity(L=torch.tensor(L, **tkwargs),
                                    VOLTAGE=torch.tensor(voltage, **tkwargs),
                                    PHI0=torch.tensor(phi0, **tkwargs),
                                    RF_FREQUENCY=torch.tensor(rf, **tkwargs),
                                    TILT=torch.tensor(tilt, **tkwargs))
        
        p_out = track_a_crab_cavity_torch(
            Particle(*self.tvec1, self.ts, self.tp0c, self.tmc2),
            cav_tilt)
        
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_crab_cavity_tilt.bmad -noplot')
        
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_cav = lambda x: track_a_crab_cavity_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), cav_tilt)[:6]
        
        J = jacobian(f_cav, self.tvec1)
        mat_py = torch.vstack(J)
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'], **tkwargs)
        if torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14) == False:
            warnings.warn('Jacobian not the same as Bmad to double precission')
            # At least should be the same up to single precission
            assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-8)
            
    def test_rf_cavity(self):
        # no offset nor tilt
        # test one particle tracking
        L = 0.2 #Length in m
        voltage = 1.0e4 #Quad focusing strength. Positive is focusing in x
        phi0 = 0.5
        rf = 1.0e9
        cav = torch_rf_cavity(L=torch.tensor(L, **tkwargs),
                              VOLTAGE=torch.tensor(voltage, **tkwargs),
                              PHI0=torch.tensor(phi0, **tkwargs),
                              RF_FREQUENCY=torch.tensor(rf, **tkwargs))
        
        p_out = track_a_rf_cavity_torch(self.p_in, cav)
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao('-lat tests/bmad_lattices/test_rf_cavity.bmad -noplot')
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_cav = lambda x: track_a_rf_cavity_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), cav)[:6]
        
        J = jacobian(f_cav, self.tvec1)
        mat_py = torch.vstack(J)
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
    
    def test_rf_cavity_offset(self):
        # x-y offset, no tilt
        # test one particle tracking
        L = 0.2
        voltage = 1.0e3
        phi0 = 0.5
        rf = 1.0e9
        x_off = 1e-3
        y_off = -2e-3
        cav_off = torch_rf_cavity(L=torch.tensor(L, **tkwargs),
                                  VOLTAGE=torch.tensor(voltage, **tkwargs),
                                  PHI0=torch.tensor(phi0, **tkwargs),
                                  RF_FREQUENCY=torch.tensor(rf, **tkwargs),
                                  X_OFFSET=torch.tensor(x_off, **tkwargs),
                                  Y_OFFSET=torch.tensor(y_off, **tkwargs))

        p_out = track_a_rf_cavity_torch(
            Particle(*self.tvec1, self.ts, self.tp0c, self.tmc2),
            cav_off)
        
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_rf_cavity_offset.bmad -noplot')
        
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_cav = lambda x: track_a_rf_cavity_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), cav_off)[:6]
        
        J = jacobian(f_cav, self.tvec1)
        mat_py = torch.vstack(J)
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
        
    def test_rf_cavity_tilt(self):
        # tilt, no offset
        # test one particle tracking
        L = 0.2
        voltage = 1.0e3
        phi0 = 0.5
        rf = 1.0e9
        tilt = 0.3
        cav_tilt = torch_rf_cavity(L=torch.tensor(L, **tkwargs),
                                   VOLTAGE=torch.tensor(voltage, **tkwargs),
                                   PHI0=torch.tensor(phi0, **tkwargs),
                                   RF_FREQUENCY=torch.tensor(rf, **tkwargs),
                                   TILT=torch.tensor(tilt, **tkwargs))
        
        p_out = track_a_rf_cavity_torch(
            Particle(*self.tvec1, self.ts, self.tp0c, self.tmc2),
            cav_tilt)
        
        x_py = torch.hstack(p_out[:6]).detach()
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_rf_cavity_tilt.bmad -noplot')
        
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=1)
        
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_cav = lambda x: track_a_rf_cavity_torch(
            Particle(*x, self.ts, self.tp0c, self.tmc2), cav_tilt)[:6]
        
        J = jacobian(f_cav, self.tvec1)
        mat_py = torch.vstack(J)
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'], **tkwargs)
        if torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14) == False:
            warnings.warn('Jacobian not the same as Bmad to double precission')
            # At least should be the same up to single precission
            assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-8)
            
        
    def test_lattice(self):
        # Create drift
        L_d = 1.0 # Drift length in m
        d1 = Drift(torch.tensor(L_d, **tkwargs))
        # Create quad
        L_q = 0.1  # quad length in m
        K1 = 10  # Quad focusing strength. Positive is focusing in x
        q1 = torch_quadrupole(L=torch.tensor(L_q, **tkwargs),
                              K1=torch.tensor(K1, **tkwargs))
        # Lattice
        lattice = [d1, q1, d1, q1, d1]  # lattice is a list of elements
        # List of particle coordinates after each element:
        x_list = [torch.hstack(coords[:6]).detach()
                  for coords in track_a_lattice(self.p_in, lattice)]
        # Outgoing particle after complete lattice:
        x_py = torch.hstack(
            track_a_lattice(self.p_in, lattice)[-1][:6]).detach()
        # Bmad lattice to compare
        tao = Tao('-lat tests/bmad_lattices/test_drift_quad.bmad -noplot')
        set_tao(tao, self.pvec1)
        orbit_out = tao.orbit_at_s(ele=5)
        x_tao = torch.tensor([orbit_out['x'],orbit_out['px'],
                              orbit_out['y'],orbit_out['py'],
                              orbit_out['z'],orbit_out['pz']],**tkwargs)
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Taylor map
        f_driftquadrupole = lambda x: track_a_lattice(
            Particle(*x,self.ts, self.tp0c, self.tmc2), lattice)[-1][:6]
        
        J = jacobian(f_driftquadrupole, self.tvec1)
        mat_py = torch.vstack(J)
        # Bmad Jacobian
        lat_tao = tao.matrix(0,5)
        mat_tao = torch.tensor(lat_tao['mat6'], **tkwargs)
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)

    def test_hessian(self):
        # Particle bunch with Gaussian distribution
        sample_size = 1000
        mean = torch.zeros(6, **tkwargs)
        cov = torch.diag(torch.tensor([1e-6, 2e-6, 1e-6, 
                                       2e-6, 1e-6, 2e-6],**tkwargs))
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mean, cov)
        sample = dist.sample(torch.Size([sample_size]))
        p_in = Particle(*sample.T, self.ts, self.tp0c, self.tmc2)
        L_d = 1.00 # Drift length
        L_q = 0.1 # Quad length 
        drift = Drift(torch.tensor(L_d, **tkwargs))

        def sigmax_end(k1s):
            """returns x beamsize after lattice composed by len(k1s)+1 
            drifts with len(k1s) quadrupoles in between.
            """
            lattice = [drift]

            for k1 in k1s:
                lattice.append(torch_quadrupole(L=torch.tensor(L_q,
                                                               **tkwargs),
                                                K1=k1))
                
                lattice.append(drift)

            p_out = track_a_lattice(p_in, lattice)[-1]
            return torch.std(p_out.x)
        
        k1s = torch.zeros(5, **tkwargs)
        hessian_py = hessian(sigmax_end,k1s)
        isnan = torch.isnan(hessian_py)
        assert torch.any(~isnan)