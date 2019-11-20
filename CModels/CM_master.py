# -*- coding: utf-8 -*-

import numpy as np
import meshio as io
import os
import time
import cfast
from cond_classes import MicroScale

class CM_model:   

    def __init__(self, params, flags):    
        
#        save_vtk, logfile, break_when_cv, reverse, verbose = flags
        
        self.ncells = params.get('ncells')
        self.ndiv = params.get('ndiv')
        self.dx = params.get('dx')
        self.dt = params.get('dt')
        self.gjc = params.get('gjc')
        self.vtkpath = params.get('vtkpath')
        self.logpath = params.get('logpath')
        self.fname = params.get('fname')
        
        self.ep_params = params.get('nvar'), params.get('nalg'), params.get('nconsts')
        self.dyn_params = params.get('sigg'), params.get('kg')
        self.cond_params = (params.get('Am'), params.get('Cm'), params.get('rho_myo'), 
                            params.get('rd0'), params.get('Acell'), params.get('eta'))
        self.time_params = (params.get('Tf'), params.get('period'), params.get('tdur'))
        self.gam_params = np.asarray((params.get('Am'), params.get('Cm'), 
                            params.get('rho_myo'), params.get('rd0'), 
                            params.get('dx'), params.get('dt'), self.gjc*params.get('eta')))
        
        if flags[0]:
            if not os.path.exists(self.vtkpath):
                os.mkdir(self.vtkpath)
                
        if flags[1]:
            if not os.path.exists(self.logpath):
                os.mkdir(self.logpath)
        
        self.rev_int = int(flags[3])
        self.flags = flags
        
        # init connexin  
        self.Cx = MicroScale(params['Cx'])
        self.Cxparams = np.asarray(self.Cx.get_params())
        
        self.model = params['model']
        
        self.params = params
        self.cv = []
        self.APD = []
        
        if params['tcond'] == 'ss':  # steady state conduction
            self.tcond = 1
        else:                       # instantaneous conduction
            self.tcond = 0  
            
        
    
    def set_mesh(self):
        dx = self.dx
        ncells = self.ncells
        ndiv = self.ndiv
        n = ncells*(ndiv) + 2   # add 2 for flux boundary conditions at 
                                # the beginning and the endof the strand
        L = self.params['L']
        xcoord = np.arange(0., L, dx)+dx/2.
                                   
        # coordinates and connectivity to plot          
        xyz = np.zeros([n-2, 3])
        xyz[:,0] = xcoord
        ien = np.vstack([np.arange(0,n-3),np.arange(0,n-3)+1]).T 
    
        
        self.cells = {'line': ien}
        self.xyz = xyz
        self.xcoord = xcoord
        self.ndiv = ndiv
        self.n = n
        
        
    def update_gjnorm(self, Vj, gjnorm_old):    # for dynamics gaps
        sigg, kg = self.dyn_params
        taug = (sigg*(np.exp(kg*np.abs(Vj))))**(-1)
        
        gjnorm_ss = self.Cx.gjnorm(Vj, self.params['tcond'])
        
        gjnorm = gjnorm_old + self.dt*(gjnorm_ss - gjnorm_old)/taug
            
        return gjnorm, gjnorm_ss, taug
            
    
    def update_gamma(self):
        Am, Cm, rho_myo, rd0, Acell, eta = self.cond_params
        sigg, kg = self.dyn_params
        
        dx = self.dx
        dt = self.dt
        gamma = self.gamma
        gappos = self.gappos
        ndiv = self.ndiv
        
        factor = eta*self.gjc
        
        u = self.u
        gjnorm = self.gjnorm
        condtype = self.condtype
        for i in gappos:            # only update voltage dependant gaps     
            if condtype[i] == 2:   # Gap - Non Ohmic
                Vj = u[i+1+(ndiv-1)] - u[i-(ndiv-1)]
                gjnorm[i] = self.Cx.gjnorm(Vj) 
                
            elif condtype[i] == 3:   # Gap - Non Ohmic
                Vj = u[i+1+(ndiv-1)] - u[i-(ndiv-1)]
                gjnorm[i], self.g_ss[i], self.taug[i] = self.update_gjnorm(Vj, gjnorm[i]) 
                
            rd = rd0/(factor*gjnorm[i])
            sig = 1./(rho_myo+rd/dx)
            gamma[i] = (dt*sig)/(Am*Cm*dx**2)
            
            
            
    def init_gamma(self):
        # identify where are the gaps
        n = self.n
        ndiv = self.ndiv
        cond_type = self.params['condtype']
        
        gappos = np.arange(ndiv, n-2, ndiv)    # pointer to gaps
        
        condtype = np.zeros(n-1, dtype = int)  # 0: intracellular ohmic
        if cond_type == 'O': # Ohmic
            condtype[gappos] = 1   
        elif cond_type == 'NO': # Non Ohmic
            condtype[gappos] = 2    
        elif cond_type == 'NOD': # Non Ohmic Dynamic
            condtype[gappos] = 3   
        
        self.gappos = gappos
        self.condtype = condtype
        
        # initiate gamma
        Am, Cm, rho_myo, rd0, Acell, eta = self.cond_params
        dx = self.dx
        dt = self.dt        
        u = self.u
        
        gamma = np.zeros(len(condtype))
        gjnorm = np.zeros(len(condtype))
        g_ss = np.zeros(len(condtype))
        taug = np.zeros(len(condtype))
        factor = eta*self.gjc
        for i in range(len(condtype)):
            if condtype[i] == 0:    # intracellular
                sig = 1./rho_myo
                gjnorm[i] = rho_myo*dx/Acell
                
            elif condtype[i] == 1:   # Gap - Ohmic
                rd = rd0/factor
                sig = 1./(rd/dx)
                
            elif condtype[i] == 2:   # Gap - Non Ohmic
                Vj = u[i+1+(ndiv-1)] - u[i-(ndiv-1)]
                gjnorm[i] = self.Cx.gjnorm(Vj, self.params['tcond']) 
                
                rd = rd0/(factor*gjnorm[i])
                sig = 1./(rd/dx)
                
            elif condtype[i] == 3:   # Gap - Non Ohmic
                Vj = u[i+1+(ndiv-1)] - u[i-(ndiv-1)]
                gjnorm[i] = self.Cx.gjnorm(Vj, 'ss') 
                _, g_ss[i], taug[i] = self.update_gjnorm(Vj, gjnorm[i]) 
                
                rd = rd0/(factor*gjnorm[i])
                sig = 1./(rd/dx)
            
            gamma[i] = (dt*sig)/(Am*Cm*dx**2)
                
        self.gamma = gamma
        self.gjnorm = gjnorm
        self.g_ss = g_ss
        self.taug = taug
        self.ngap = len(gappos)
    
    
         
            
        
    def set_monitor_nodes(self, cell1 = 22, cell2 = 42, cell3 = 32, cv_lim = 30.):
        ndiv = self.ndiv
        
        n1 = int((cell1-1)*(ndiv) + 1)
        n2 = int((cell2-1)*(ndiv) + 1)
        n3 = int((cell3-1)*(ndiv) + 1)
        x1 = self.xcoord[n1]    
        x2 = self.xcoord[n2]  
        x3 = self.xcoord[n3]   
        if self.flags[3]:   # if reverse
            n1a = n1
            n1 = n2
            n2 = n1a      
    
        self.cv_lim = cv_lim
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        
           
        
         
    def set_init_conditions(self):
        nvar, nalg, nconsts = self.ep_params
        n = self.n
        
        # Initialize Variables
        consts = np.zeros([nconsts])
        states = np.zeros([nvar])
        cfast.c_initConsts(consts, states) # get initial conditions for EP Model
        self.consts = consts
        
        # transmembrane potential
        u = np.ones(n)*states[0]        # t d ransmembrane voltage
        
        self.u_n = np.array(u)
        self.u = u
            
        # gating variables
        r_n = np.squeeze(np.repeat(np.array([[states[1::]]]), n, axis=1))
                
        self.r_n = r_n
        self.r = np.array(r_n)
        
        # constants for finite differences
        self.init_gamma()
        
        
        
    def set_flux(self):
        qb, rho_myo, RCG = self.params['source'], self.params['rho_myo'], self.params['RCG']
        source = 2*self.dx*qb*(rho_myo*RCG)
        
        self.u_n[0] = self.u_n[2] - source
        self.source = source
        
        
    def save_vtk(self, cont):                
        p_data = {'Vm' : self.u[1:-1], 'g': self.gjnorm[0:-1], 
                  'taug': self.taug[0:-1], 'gjnorm': self.g_ss[0:-1], 'r': self.r[1:-1,:]}
        io.write_points_cells(self.vtkpath + self.fname + '_%06i' % cont+".vtu", self.xyz, 
                              self.cells, point_data = p_data)
        
        
    def control_CV(self, t):
        n1, n2 = self.n1, self.n2
        x1, x2 = self.x1, self.x2
        
        ua = self.u
        ua_n = self.u_n
        cv_lim = self.cv_lim
        dt = self.dt
        
        if ua[n1]>cv_lim and ua_n[n1]<cv_lim:
            self.act_time1 = (t-dt) + dt/(ua[n1]-ua_n[n1])*(cv_lim-ua_n[n1])
        if ua[n2]>cv_lim and ua_n[n2]<cv_lim:
            self.act_time2 = (t-dt) + dt/(ua[n2]-ua_n[n2])*(cv_lim-ua_n[n2])
            self.cv.append((x2-x1)/(self.act_time2 - self.act_time1)*100)

        
        

    def sanity_checks(self):
        if np.isnan(max(self.u)):
            raise ValueError('NaN value in Vm')
            
            
    def write_log(self, t):
        rho_myo = self.params['rho_myo']
        rd = self.params['rd0']
        
        textfile = open(self.logpath+'/L'+self.fname,"w")
        textfile.write('1D - Electrophysiology Simulation'+'\n')
        textfile.write('Conductivity Type: Ohmic'+'\n')
        textfile.write('\n')
        textfile.write('Ionic Model: Luo Rudy (1991)'+'\n')
        textfile.write('Source: '+ str(self.source)+' mV/ms'+'\n')
        textfile.write('Mesh Size: '+ str(self.dx)+' mm'+'\n')
        textfile.write('dt: '+ str(self.dt)+' ms'+'\n')
        textfile.write('rho_myo: '+ str(rho_myo) + ' Ohm-m'+'\n')
        textfile.write('r_d: '+ str(rd) + ' Ohm-m-mm'+'\n')
        textfile.write('Final Time: '+ str(t)+' ms'+'\n')
        textfile.write('\n')
        textfile.write('----------------------------------------'+'\n')
        textfile.write('RESULTS'+'\n')    
        textfile.write('Simulation Time: ' + str(self.el_time) + ' s'+'\n')
        textfile.write('CV: '+str(self.cv)+' cm/s'+'\n')
        textfile.close()        
        
    
    def solve(self):
        dt = self.dt
        cond_type = self.params['condtype']
        Tf, period, tdur = self.time_params
        save_vtk, logfile, break_when_cv, reverse, verbose = self.flags
        nvar, nalg, _ = self.ep_params
        
        # set time interval
        voi = np.linspace(0, Tf, int(Tf/dt)+1)
        cum_dt = 0.
        dt_save = self.params['dtsave']
        
        # save initial states
        cont = 0
        
        if save_vtk:
            self.save_vtk(cont)
            cont += 1
                
        source = self.source
        
        u = self.u
        un = self.u_n
        r = self.r
        rn = self.r_n
        
        wct_start = time.time()
        cycle = 0
        # explicit time-stepping loop
        for i,t in enumerate(voi[1::]):
            # Applied source for a certain time
            if int(t/period) > cycle:
                print(self.model + '_gjc' + str(self.gjc) + ' cycle: ' + str(cycle) + '; CV: ' + str(self.cv))
            cycle = int(t/period)
            if t > (period*cycle) and t < (period*cycle+tdur):
                source1 = source
            else:
                source1 = 0.
                
            # Conductivity update
            if cond_type == 'NO':
                cfast.c_NOupdate(un, self.gappos, self.Cxparams, self.gam_params,
                                 self.ndiv, self.ngap, self.tcond, self.gjnorm, self.gamma)
            elif cond_type == 'NOD':
                cfast.c_NODupdate(un, self.gappos, np.asarray(self.dyn_params), 
                                  self.Cxparams,self.gam_params, self.ndiv, self.ngap, 
                                 self.gjnorm, self.g_ss, self.taug, self.gamma)
                
            # Finite difference update
            cfast.c_FDupdate(un, rn, self.consts, t, dt, self.n, nvar,
                             nalg, self.gamma, source1, self.rev_int, u, r)

            # control CV
            self.control_CV(t)
            if len(self.cv) == break_when_cv:
                if logfile:
                    self.el_time = time.time() - wct_start
                    self.write_log(t) 
                break
                                  
            # sanity checks
            self.sanity_checks()
            
            # save vtk
            cum_dt += dt  
            if (np.round(cum_dt,6) >= dt_save):
                if verbose:
                    print(self.fname,'output at t = ', t)
                    print(max(u))
                if save_vtk:
                    self.save_vtk(cont)
                    cont += 1
                cum_dt = 0.  
    
            # next loop
            un = np.array(u)
            rn = np.array(r)
            self.u_n = un
            self.r_n = rn
            
            
        # Finish simulation
        sim_time = t
        self.el_time = time.time() - wct_start
        
        if verbose:
            print("\nTotal time steps = %d \nTotal simulation time = %5.1f ms \nTotal computing time = %8.1f seconds" 
                  % (len(voi)-1,sim_time, self.el_time))
            
            print('GJ coupling = ' + str(self.gjc*100) 
                                                + ', CV = ' + str(self.cv))
            
    def run(self):
        self.set_mesh()
        self.set_monitor_nodes(cell1 = 22, cell2 = 42, cv_lim = -30.)
        self.set_init_conditions()
        self.set_flux()
        self.solve()
            
            

if __name__ == "__main__":
    
    from Parameters import Parameters
    
    # Flags
    save_vtk = True
    logfile = True
    break_when_cv = 1       
    reverse = False
    verbose = True
    
    flags = save_vtk, logfile, break_when_cv, reverse, verbose

    dm_params = Parameters('DNOM')
    dm_params['tcond'] = 'in'
    dm_params['gjc'] = 100.
    dm_params['period'] = 400.
    dm_params['Tf'] = 400*10
    dm_params['dtsave'] = 0.5
    dm_params['fname'] =  dm_params['model']  + '_gjc'+str( dm_params['gjc']*100)+'_mod'  # especific file name
    model = CM_model(dm_params, flags)
    model.run()
    
    
    
    
