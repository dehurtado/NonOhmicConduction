# -*- coding: utf-8 -*-

import dolfin as d
import numpy as np
import meshio as io
import os
import time
from cond_classes import MicroScale, MacroScale
import scipy.sparse as ss
import scipy.sparse.linalg as sl
import cfast


class HM_model():   

    def __init__(self, params, flags):    
        
        self.xmax = params.get('L')
        self.dx = params.get('dx')
        self.dt = params.get('dt')
        self.gjc = params.get('gjc')
        self.vtkpath = params.get('vtkpath')
        self.logpath = params.get('logpath')
        self.fname = params.get('fname')
        
        self.ep_params = params.get('nvar'), params.get('nalg'), params.get('nconsts')
        self.dyn_params = params.get('sigg'), params.get('kg')
        self.cond_params = (params.get('Am'), params.get('Cm'), params.get('rho_myo'), 
                            params.get('eta'), params.get('k'))
        self.time_params = (params.get('Tf'), params.get('period'), params.get('tdur'))
        self.hom_params = params.get('delta'), params.get('epsilon')
        
        if flags[0]:
            if not os.path.exists(self.vtkpath):
                os.mkdir(self.vtkpath)
                
        if flags[1]:
            if not os.path.exists(self.logpath):
                os.mkdir(self.logpath)
        
        self.rev_int = int(flags[3])
        self.flags = flags
        
        self.model = params['model']
        
        # init connexin  
        self.Cx = MicroScale(params['Cx'])
        self.tcond = params['tcond']
        
        self.params = params
        
        self.cv = []
        
    def set_mesh(self):
        dx, xmax = self.dx, self.xmax
        nel = int(xmax/dx)    # Number of elements
        
        mesh = d.IntervalMesh(nel, 0, xmax)
        self.mesh = mesh
        
        # to output
        self.cells = {'line': mesh.cells()}
        xyz = mesh.coordinates()
        self.xyzio = np.zeros([xyz.shape[0], 3])
        self.xyzio[:,0] = xyz.flatten()
        
            
    def set_spaces_functions(self, p):
        # p: int, polynomial order of trial/test functions
        nvar = self.params['nvar']
        mesh = self.mesh
        
        # Define finite-element spaces
        U = d.FunctionSpace(mesh, "CG", p)
        W = d.VectorFunctionSpace(mesh, "CG", p, dim = nvar-1)
        
        # Functions for bilinear and linear form
        self.ut = d.TrialFunction(U)
        self.du = d.TestFunction(U)
        self.un = d.Function(U)
        self.u = d.Function(U)
        self.w = d.Function(W)
        
        # Auxiliar functions
        self.sig = d.Function(U) # conductivity value
        self.f = d.Function(U) # conductivity value
        
        # Save Spaces
        self.U = U
        self.W = W
        
        self.ndofs = U.tabulate_dof_coordinates().shape[0]
        
        
    def set_monitor_nodes(self, x1 = 2.2, x2 = 4.2, cv_lim = -30.):
        n1, n2 = 0.,0.
        meshV = self.U.tabulate_dof_coordinates()
        meshVl = list(meshV)
        if x1 in meshVl:
            n1 = meshVl.index(x1)
        if x2 in meshVl:
            n2 = meshVl.index(x2)
        if self.flags[3]:
            n1a = n1
            n1 = n2
            n2 = n1a  
            
        self.cv_lim = cv_lim
        self.n1 = n1
        self.n2 = n2
        self.x1 = x1
        self.x2 = x2
        
        
    def set_BC(self):
        # No dirichlet condition
        def boundary(x, on_boundary):
            return False
        
         
    def set_init_conditions(self):
        ndofs = self.ndofs
        nvar, nalg, nconsts = self.ep_params
        
        # initial conditions
        consts = np.zeros([nconsts])
        states = np.zeros([nvar])
        rates = np.zeros([nvar])
        algebraic = np.zeros([nalg])
        cfast.c_initConsts(consts, states) # get initial conditions for EP Model
        self.consts = consts
        
        
        u_D = d.Expression('x[0] < DOLFIN_EPS ? Vinit : Vinit ', degree = 2, Vinit = states[0])
        self.un = d.interpolate(u_D, self.U)
        
        self.w.vector().set_local(np.repeat(np.array([[states[1::]]]), ndofs,axis=1).flatten('C'))
    
        # initial rates
        f = d.Function(self.U)
        cfast.c_computeRates(0., consts, rates, states, algebraic)
        f.vector().set_local(np.ones(ndofs)*rates[0])
        self.f = f
    
        # compute gradient
        self.gradu = self.u.dx(0)
        
        Am, Cm, rho_myo, eta, k = self.cond_params
        delta, epsilon = self.hom_params
        factor = eta*self.gjc
    
        if self.model == 'NOM':
            sigc = 1/rho_myo
    
            # initialize homogenization class  
            macro_params = sigc, Am, Cm, factor, delta, epsilon, k
            MS = MacroScale(self.Cx, macro_params, self.tcond)
            
            rang = -1000., 1000., 201.
            self.y_arr, self.sig_arr, self.y_range, self.s_range, eta = MS.get_conductivity(rang)
                            
            sig0 = MS.get_cond0()   # conductivity value when \nabla V_m = 0
            self.sig.vector().set_local(np.ones(ndofs)*sig0) 
            
        
        elif self.model == 'LHM':
            
            rd = self.params['rd0']/factor
            sigma = 1./(rd/epsilon + rho_myo)
            
            self.sig = sigma/(Am*Cm)
            
        
        
        
    def set_flux(self): 
        Am, Cm = self.params['Am'], self.params['Cm']
        qb, tdur = self.params['source'], self.params['tdur']
        xmax = self.params['L']
        
        # Neumann condition 
        source = qb/(Am*Cm)
                       
        if self.flags[3]:
            q = d.Expression('x[0] >= L - DOLFIN_EPS ? source : 0',
                       degree = 2, t = 0, tdur = tdur, source = source, L = xmax)
            
        else:
            q = d.Expression('x[0] < DOLFIN_EPS ? source : 0',
                       degree = 2, t = 0, tdur = tdur, source = source)
                   
        self.q = q
        self.source = source
        
        
    def set_variational_form(self):
        dt = self.dt
        
        a = 1./dt*self.ut*self.du*d.dx
        self.L = (1./dt*self.un + self.f)*self.du*d.dx \
            - self.sig*d.inner(self.un.dx(0),self.du.dx(0))*d.dx - self.q*self.du*d.ds
            
        # Invert tangent matrix one time and reuse it
        K_mat = d.assemble(a)
        K_mat = d.as_backend_type(K_mat).mat()
        K = ss.csr_matrix(K_mat.getValuesCSR()[::-1], shape = K_mat.size)
        self.Kinv = sl.inv(K)

        # Assemble lineal form
        self.F_vec = d.assemble(self.L)
              
        
        
    def save_vtk(self, cont):
        ua = self.u.vector().get_local()
        sa = self.sig.vector().get_local()
        wa = self.w.vector().get_local().reshape([-1,self.params['nvar']-1])
        grad_ua = d.project(self.gradu, self.U).vector().get_local()
        
        p_data = {'Vm' : ua, 'gradu': grad_ua, 'sigma': sa, 'r': wa}
        io.write_points_cells(self.vtkpath + self.fname + '_%06i' % cont+'.vtu', self.xyzio, 
                              self.cells, point_data = p_data)
        
        
    def control_CV(self, t):
        n1, n2 = self.n1, self.n2
        x1, x2 = self.x1, self.x2
        
        ua = self.u.vector().get_local()
        ua_n = self.un.vector().get_local()
        cv_lim = self.cv_lim
        dt = self.dt
        
        if ua[n1]>cv_lim and ua_n[n1]<cv_lim:
            self.act_time1 = (t-dt) + dt/(ua[n1]-ua_n[n1])*(cv_lim-ua_n[n1])
        if ua[n2]>cv_lim and ua_n[n2]<cv_lim:
            self.act_time2 = (t-dt) + dt/(ua[n2]-ua_n[n2])*(cv_lim-ua_n[n2])
            self.cv.append((x2-x1)/(self.act_time2-self.act_time1)*100)
            print(self.cv)   


    def sanity_checks(self):
        ua = self.u.vector().get_local()
        if np.isnan(max(ua)):
            raise ValueError('NaN value in Vm')
            
            
    def write_log(self, t):
        
        textfile = open(self.logpath+'/L' + self.fname,"w")
        textfile.write('1D - Electrophysiology Simulation'+'\n')
        textfile.write('\n')
        textfile.write('Ionic Model: Luo Rudy (1991)'+'\n')
        textfile.write('Conductivity Type: Non-Ohmic'+'\n')
        textfile.write('\n')
        textfile.write('Source: '+ str(self.source)+' microA_per_cm2'+'\n')
        textfile.write('Mesh Size: '+ str(self.dx)+' mm'+'\n')
        textfile.write('dt: '+ str(self.dt)+' ms'+'\n')
        textfile.write('Final Time: '+ str(t)+' ms'+'\n')
        textfile.write('GJ coupling: '+ str(self.gjc*100.)+' %'+'\n')
        textfile.write('\n')
        textfile.write('----------------------------------------'+'\n')
        textfile.write('RESULTS'+'\n')    
        textfile.write('Simulation Time: ' + str(self.el_time) + ' s'+'\n')
        textfile.write('CV: ' + str(self.cv)+' cm/s'+'\n')
        textfile.close()        
        
    
    def solve(self):
        Tf, period, tdur = self.time_params
        dt = self.dt
        
        # set time interval
        voi = np.linspace(0, Tf, int(Tf/dt)+1)
        cum_dt = 0.
        dt_save = self.params['dtsave']
        
        # save initial states
        cont = 0
        
        if self.flags[0]:
            self.save_vtk(cont)
            cont += 1
               
        q = self.q
        
        if self.model == 'NOM':
            sa = self.sig.vector().get_local()
        fa = self.f.vector().get_local()
        wct_start = time.time()
        cycle = 0
        
        self.el_time = 0
        # explicit time-stepping loop
        for i,t in enumerate(voi[1::]):
            # Applied source for a certain time
            if int(t/period) > cycle:
                print(self.model + '_gjc' + str(self.gjc) + ' cycle: ' + str(cycle) + '; CV: ' + str(self.cv))
                
            cycle = int(t/period)
            if t > (period*cycle) and t < (period*cycle+tdur):
                q.source = self.source
            else:
                q.source = 0
                
            # update time
            cum_dt += dt  
            q.t = t
            
            if self.model == 'NOM': 
                # Compute conductivity
                gradu = d.project(self.gradu, self.U)
                grad_ua = gradu.vector().get_local()
    #            print(np.max(self.sig.vector().get_local()))
                cfast.c_effcond(grad_ua, self.y_arr, self.sig_arr, self.y_range, 
                                self.s_range, len(grad_ua), sa) 
                self.sig.vector().set_local(sa)
                self.sig.vector().apply('insert')
                
            # solve variational problem for u   
            d.assemble(self.L, tensor = self.F_vec)  
            F = self.F_vec.get_local()
            uss = self.Kinv*F
            self.u.vector().set_local(uss)    
            
            # solve ODE at nodes of gating variable
            ua_n = self.un.vector().get_local()
            wa = self.w.vector().get_local()
            
            cfast.c_update(ua_n, wa, self.consts, t, dt, self.ndofs, fa)
            self.f.vector().set_local(fa)
            self.w.vector().set_local(wa)
                        
            # control CV
            self.control_CV(t)
            if len(self.cv) == self.flags[2]:
                if self.flags[1]:
                    self.write_log(t) 
                break
            
            # sanity checks
            self.sanity_checks()
            # save vtk
            if (np.round(cum_dt,6) >= dt_save):
                if self.flags[4]:
                    print(self.fname,'output at t = ', t)
                    print(np.max(self.u.vector().get_local()))
                    
                if self.flags[0]:
                    self.save_vtk(cont)
                    cont += 1
                cum_dt = 0.        
                
            # update variables
            self.un.assign(self.u)
            
            
        # Finish simulation
        sim_time = t
        self.el_time = time.time() - wct_start
        
        if self.flags[4]:
            print("\nTotal time steps = %d \nTotal simulation time = %5.1f ms \nTotal computing time = %8.1f seconds" % (len(voi)-1,sim_time, self.el_time))
            
            print('GJ coupling = ' + str(self.gjc*100) 
                                                + ', CV = ' + str(self.cv))
            
    def run(self):
        self.set_mesh()
        self.set_spaces_functions(1)
        self.set_monitor_nodes()
        self.set_init_conditions()
        self.set_flux()
        self.set_variational_form()
        self.solve()
            
            

if __name__ == "__main__":
    
    from Parameters import Parameters
    
    # Flags
    save_vtk = False
    logfile = False
    break_when_cv = 1   
    reverse = False
    verbose = True
    
    flags = save_vtk, logfile, break_when_cv, reverse, verbose

    hm_params = Parameters('NOM')
    hm_params['gjc'] = 100.
    hm_params['Tf'] = 10000.
    hm_params['period'] = 300.
    hm_params['dtsave'] = 0.5
    hm_params['tcond'] = 'in'
    hm_params['source'] = -25
    hm_params['k'] = 2*hm_params['epsilon']
    hm_params['fname'] = hm_params['model'] + '_gjc' + str(hm_params['gjc']*100)  +'mod'
    
    
    model = HM_model(hm_params, flags)
    model.run()
    
