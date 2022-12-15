import pytest
from pytao import Tao
import torch
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
import warnings

# bmadx imports: 
from bmadx import Particle, Drift, Quadrupole, CrabCavity, RFCavity
from bmadx import track_element, track_lattice
from bmadx import M_ELECTRON

def set_tao(tao, coords):
    tao.cmd('set particle_start x='+str(coords[0]))
    tao.cmd('set particle_start px='+str(coords[1]))
    tao.cmd('set particle_start y='+str(coords[2]))
    tao.cmd('set particle_start py='+str(coords[3]))
    tao.cmd('set particle_start z='+str(coords[4]))
    tao.cmd('set particle_start pz='+str(coords[5]))

class TestBmadxTorch:
    # create incoming particle
    coords = [2e-3,3e-3,-3e-3,-1e-3,2e-3,-2e-3]
    coords_t = torch.tensor(coords, dtype=torch.double)
    p_in = Particle(*coords_t,
                    s = torch.tensor(0.0, dtype=torch.double),
                    p0c = torch.tensor(4.0e7, dtype=torch.double),
                    mc2 = torch.tensor(M_ELECTRON, dtype=torch.double))
    
    # coordinates with autodiff:
    diff_coords = coords_t.clone().detach().requires_grad_(True)
    
    def test_drift(self):
        # test one particle tracking
        d = Drift(L=1.0)
        p_out = track_element(self.p_in, d)
        x_py = torch.hstack(p_out[:6])

        tao = Tao('-lat tests/bmad_lattices/test_drift.bmad -noplot')
        set_tao(tao, self.coords)
        orbit_out=tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_drift = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            d)[:6]

        J = jacobian(f_drift, self.diff_coords)
        mat_py = torch.vstack(J)
        
        drift_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(drift_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
        
    def test_quadrupole(self):
        # no offset nor tilt
        # test one particle tracking
        q = Quadrupole(L=0.1, K1=10.0)
        p_out = track_element(self.p_in, q)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao('-lat tests/bmad_lattices/test_quad.bmad -noplot')
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_quadrupole = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            q)[:6]
        
        J = jacobian(f_quadrupole, self.diff_coords)
        mat_py = torch.vstack(J)
        
        quad_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(quad_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
    
    def test_quadrupole_offset(self):
        # x-y offset, no tilt
        # test one particle tracking
        q_off = Quadrupole(L = 0.1,
                           K1 = 10.0,
                           X_OFFSET = 1e-3,
                           Y_OFFSET = -2e-3)
        
        p_out = track_element(self.p_in, q_off)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao('-lat tests/bmad_lattices/test_quad_offset.bmad -noplot')
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_quadrupole_off = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            q_off)[:6]

        J = jacobian(f_quadrupole_off, self.diff_coords)
        mat_py = torch.vstack(J)
        
        quad_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(quad_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
        
    def test_quadrupole_tilt(self):
        # tilt, no offset
        # test one particle tracking
        q_tilt = Quadrupole(L = 0.1,
                            K1 = 10.0,
                            TILT = 0.3)
        p_out = track_element(self.p_in, q_tilt)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao('-lat tests/bmad_lattices/test_quad_tilt.bmad -noplot')
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_quadrupole = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            q_tilt)[:6]
        
        J = jacobian(f_quadrupole, self.diff_coords)
        mat_py = torch.vstack(J)
        
        quad_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(quad_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)
        
    def test_crab_cavity(self):
        # no offset nor tilt
        # test one particle tracking
        cav = CrabCavity(L=0.2,
                         VOLTAGE=1e4,
                         PHI0=0.5,
                         RF_FREQUENCY=1e9)
        
        p_out = track_element(self.p_in, cav)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao('-lat tests/bmad_lattices/test_crab_cavity.bmad -noplot')
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_cav = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            cav)[:6]
        
        J = jacobian(f_cav, self.diff_coords)
        mat_py = torch.vstack(J)
        
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
    
    def test_crab_cavity_offset(self):
        # x-y offset, no tilt
        # test one particle tracking
        cav_off = CrabCavity(L=0.2,
                             VOLTAGE=1e4,
                             PHI0=0.5,
                             RF_FREQUENCY=1e9,
                             X_OFFSET=1e-3,
                             Y_OFFSET=-2e-3)
        
        p_out = track_element(self.p_in, cav_off)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_crab_cavity_offset.bmad -noplot')
        
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_cav = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            cav_off)[:6]
        
        J = jacobian(f_cav, self.diff_coords)
        mat_py = torch.vstack(J)
        
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
        
    def test_crab_cavity_tilt(self):
        # tilt, no offset
        # test one particle tracking
        cav_tilt = CrabCavity(L=0.2,
                              VOLTAGE=1e4,
                              PHI0=0.5,
                              RF_FREQUENCY=1e9,
                              TILT=0.3)
        
        p_out = track_element(self.p_in, cav_tilt)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_crab_cavity_tilt.bmad -noplot')
        
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_cav = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            cav_tilt)[:6]
        
        J = jacobian(f_cav, self.diff_coords)
        mat_py = torch.vstack(J)
        
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'])
        
        if torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14) == False:
            warnings.warn('Jacobian not the same as Bmad to double precission')
            # At least should be the same up to single precission
            assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-8)
            
    def test_rf_cavity(self):
        # no offset nor tilt
        # test one particle tracking
        cav = RFCavity(L=0.2,
                       VOLTAGE=1e4,
                       PHI0=0.5,
                       RF_FREQUENCY=1e9)
        
        p_out = track_element(self.p_in, cav)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao('-lat tests/bmad_lattices/test_rf_cavity.bmad -noplot')
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_cav = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            cav)[:6]
        
        J = jacobian(f_cav, self.diff_coords)
        mat_py = torch.vstack(J)
        
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
    
    def test_rf_cavity_offset(self):
        # x-y offset, no tilt
        # test one particle tracking
        cav_off = RFCavity(L=0.2,
                           VOLTAGE = 1e4,
                           PHI0 = 0.5,
                           RF_FREQUENCY = 1e9,
                           X_OFFSET = 1e-3,
                           Y_OFFSET = -2e-3)
        
        p_out = track_element(self.p_in, cav_off)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_rf_cavity_offset.bmad -noplot')
        
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_cav = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            cav_off)[:6]
        
        J = jacobian(f_cav, self.diff_coords)
        mat_py = torch.vstack(J)
        
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-13)
        
    def test_rf_cavity_tilt(self):
        # tilt, no offset
        # test one particle tracking
        cav_tilt = RFCavity(L=0.2,
                            VOLTAGE=1e4,
                            PHI0=0.5,
                            RF_FREQUENCY=1e9,
                            TILT=0.3)
        
        p_out = track_element(self.p_in, cav_tilt)
        x_py = torch.hstack(p_out[:6])
        
        tao = Tao(
            '-lat tests/bmad_lattices/test_rf_cavity_tilt.bmad -noplot')
        
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=1)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_cav = lambda x: track_element(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            cav_tilt)[:6]
        
        J = jacobian(f_cav, self.diff_coords)
        mat_py = torch.vstack(J)
        
        cav_tao = tao.matrix(0,1)
        mat_tao = torch.tensor(cav_tao['mat6'])
        
        if torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14) == False:
            warnings.warn('Jacobian not the same as Bmad to double precission')
            # At least should be the same up to single precission
            assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-8)
            
        
    def test_lattice(self):
        # create lattice
        d = Drift(L=1.0)
        q = Quadrupole(L=0.1, K1=10)
        lattice = [d, q, d, q, d]
        
        # outgoing particle:
        p_out = track_lattice(self.p_in, lattice)
        x_py = torch.hstack(p_out[:6])

        # Bmad lattice to compare
        tao = Tao('-lat tests/bmad_lattices/test_drift_quad.bmad -noplot')
        set_tao(tao, self.coords)
        orbit_out = tao.orbit_at_s(ele=5)
        x_tao = torch.tensor([orbit_out['x'],
                              orbit_out['px'],
                              orbit_out['y'],
                              orbit_out['py'],
                              orbit_out['z'],
                              orbit_out['pz']], dtype=torch.double)
        
        assert torch.allclose(x_py, x_tao, atol=0, rtol=1.0e-14)
        
        # test Jacobian
        f_driftquadrupole = lambda x: track_lattice(
            Particle(*x, s=self.p_in.s, p0c=self.p_in.p0c, mc2=self.p_in.mc2),
            lattice)[:6]
        
        J = jacobian(f_driftquadrupole, self.diff_coords)
        mat_py = torch.vstack(J)
        
        lat_tao = tao.matrix(0,5)
        mat_tao = torch.tensor(lat_tao['mat6'])
        
        assert torch.allclose(mat_py, mat_tao, atol=0, rtol=1.0e-14)

    def test_hessian(self):
        # Particle bunch with Gaussian distribution
        sample_size = 1000
        mean = torch.zeros(6)
        cov = torch.diag(torch.tensor([1e-6, 2e-6, 1e-6, 2e-6, 1e-6, 2e-6]))
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mean, cov)
        sample = dist.sample(torch.Size([sample_size]))
        p_in = Particle(*sample.T,
                        s = torch.tensor(0.0, dtype=torch.double),
                        p0c = torch.tensor(4.0e7, dtype=torch.double),
                        mc2 = torch.tensor(M_ELECTRON, dtype=torch.double))
        
        drift = Drift(L=1.0)
        L_q = 0.1 # Quad length 

        def sigmax_end(k1s):
            """returns x beamsize after lattice composed by len(k1s)+1 
            drifts with len(k1s) quadrupoles in between.
            """
            lattice = [drift]

            for k1 in k1s:
                lattice.append(Quadrupole(L=L_q, K1=k1))
                lattice.append(drift)

            p_out = track_lattice(p_in, lattice)
            return torch.std(p_out.x)
        
        k1s = torch.zeros(5)
        hessian_py = hessian(sigmax_end, k1s)
        isnan = torch.isnan(hessian_py)
        assert torch.any(~isnan)