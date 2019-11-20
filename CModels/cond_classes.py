# -*- coding: utf-8 -*-

import numpy as np

class MicroScale():
    def __init__(self, cx_type):
        self.connexin = cx_type

        if self.connexin == 'Cx43-Cx43':
            self.vj0_neg = -60.8
            self.vj0_pos = 62.9
            self.gjmin_neg = 0.26
            self.gjmin_pos = 0.25
            self.A_neg = -3.4/(25.7)#0.04
            self.A_pos = 2.9/(25.7)#0.04
            self.p = 1
            self.d = 0
            self.Gjn = 1.99
            self.Gjp = 2.01
            self.VHn = -175.8
            self.VHp = 175.7
            
        elif self.connexin== 'Cx45-Cx45':
            self.vj0_neg = -38.9
            self.vj0_pos = 38.5
            self.gjmin_neg = 0.17
            self.gjmin_pos = 0.16
            self.A_neg = -2.5/(25.7)#0.04
            self.A_pos = 2.7/(25.7)#0.04
            self.p = 1
            self.d = 0
            self.Gjn = 1.99
            self.Gjp = 2.02
            self.VHn = -112.7
            self.VHp = 135.
            
        elif self.connexin == 'Cx43-Cx45': # impaired
            self.vj0_neg = -15.9
            self.vj0_pos = 149.3
            self.gjmin_neg = 0.05
            self.gjmin_pos = 0.05
            self.A_neg = -2.1/(25.7)#0.04
            self.A_pos = 0.7/(25.7)#0.04
            self.p = 0.73
            self.d = 25.08
            self.Gjn = 1.93
            self.Gjp = 2.0
            self.VHn = -130.
            self.VHp = 404.
            
        else:
            print('No connexin name or unknown connexin specified, \
                  please set parameters using Connexin.set_params(params)')
            
    def set_params(self, params):
        gjmin_pos, gjmin_neg, vj0_pos, vj0_neg, A_pos, A_neg = params
        self.vj0_neg = vj0_neg
        self.vj0_pos = vj0_pos
        self.gjmin_neg = gjmin_neg
        self.gjmin_pos = gjmin_pos
        self.A_neg = A_neg/(25.7)#0.04
        self.A_pos = A_pos/(25.7)#0.04
        
    def get_params(self):
        params = self.gjmin_pos, self.gjmin_neg, self.vj0_pos, self.vj0_neg,\
                 self.A_pos, self.A_neg, self.p, self.d, self.Gjn, self.Gjp, \
                 self.VHn, self.VHp
        return params
    
    def gjnorm_ss(self, Vj):
        if(Vj >= self.d):
            gGJ_norm = (1.-self.gjmin_pos)/(self.p+
                       np.exp(self.A_pos*(Vj-self.vj0_pos))) + self.gjmin_pos
        else:
            gGJ_norm = (1.-self.gjmin_neg)/(self.p+
                       np.exp(self.A_neg*(Vj-self.vj0_neg))) + self.gjmin_neg
        return gGJ_norm
        
    
    def gjnorm_inst(self, Vj):
        def fexp(Vj, VH):
            return np.exp(Vj/(VH*(1+np.exp(-Vj/VH))))
        
        if Vj>0:
            return self.Gjp/(fexp(Vj, self.VHp) + fexp(-Vj, self.VHp))
        else:
            return self.Gjn/(fexp(Vj, self.VHn) + fexp(-Vj, self.VHn))
        
        
        
    def gjnorm(self, Vj, which):
        if which == 'ss':
            return self.gjnorm_ss(Vj)
        elif which == 'in':
            return self.gjnorm_inst(Vj)
        
    
    
class MacroScale():
    def __init__(self, Connexin, macro_params, tcond):
        
        sigc, Am, Cm, factor, delta, epsilon, k = macro_params
        
        self.sigc = sigc
        self.sigg = sigc
        self.Am = Am
        self.Cm = Cm
        self.factor = factor
        self.delta = delta
        self.epsilon = epsilon
        self.params = macro_params
        self.Cx = Connexin
        self.k = k
        self.tcond = tcond

    def mua_func(self, Vj):
        return self.factor*self.Cx.gjnorm(Vj, self.tcond) - 1
        
    def Ahat(self, eta, y):
        Vj = eta*self.k
#        Vj = self.epsilon*((2-self.delta)*y - eta)
        return self.sigc*(1. + self.mua_func(Vj))/(self.sigc/self.sigg +
                    (1.-self.delta)*(1.+self.mua_func(Vj)))
        
    def fixed_point_func(self, y, eta_n):
        return  -(1. - self.delta)*(self.Ahat(eta_n,y)/self.sigc-1.)*y
        
    def effective_conductivity(self, y):   
        sigc, Am, Cm, eta, delta, epsilon, k = self.params
        
        # Parameters for fixed point iteration
        tol = 1e-12     # tolerance
        max_iter = 200  # max number of iterations
        
        eta_n = y       # Good first assumption
        error = 1.      # Initialize error
        it = 0          # Initialize iteration counter
        
        # Fixed point iteration
        while error >= tol:
            eta_n1 = self.fixed_point_func(y,eta_n)
            
            error = np.linalg.norm(eta_n1-eta_n)
            
            eta_n = eta_n1
            
            it = it + 1
            
            if it-1 == max_iter and error>tol:
                print('Warning: Fixed Point Iteration Failed')
                print('Error: ' +str(error))
                break
            
        return self.Ahat(eta_n,y), eta_n, error, it-1
        
    def get_conductivity(self, rang):
        sigc, Am, Cm, eta, delta, epsilon, k = self.params
        
        y_min, y_max, n = rang
        y_arr = np.linspace(y_min,y_max,n)
        
        sig_eff = np.zeros(len(y_arr))
        eta = np.zeros(len(y_arr))
        for i in range(len(y_arr)):
            sig_eff[i], eta[i],_,_ = self.effective_conductivity(y_arr[i])
            
        sig_eff = sig_eff/(self.Am*self.Cm)
            
        sig_min = sig_eff[0]
        sig_max = sig_eff[-1]
        
        y_range = np.array([y_min, y_max])
        s_range = np.array([sig_min, sig_max])
        
        
        return y_arr, sig_eff, y_range, s_range, eta
    
    def get_cond0(self):
        return self.effective_conductivity(0)[0]/(self.Am*self.Cm)
    
        
        
    
        
    
        
    
        
    
        
    
        
        
    
    

