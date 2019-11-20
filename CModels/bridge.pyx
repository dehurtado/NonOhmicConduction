import cython
import numpy as np
cimport numpy as np

###################################
#   EP functions
###################################
  
cdef extern void initConsts(double* consts, # consts[0:nconsts] 
                            double* states, # states[0:nvar]
                )
                
def c_initConsts(  np.ndarray[double, ndim=1, mode="c"] consts,
                np.ndarray[double, ndim=1, mode="c"] states):
                    
    return initConsts( &consts[0],
                    &states[0])


cdef extern void computeRates(double t,     # time
                              double* consts, # consts[0:nconsts] 
                              double* rates,  # rates[0:nvar]
                              double* states, # states[0:nvar]
                              double* algebraic, # algebraic[0:nalg]
                              
                )

                
def c_computeRates(  double t,    # length of arrays IN
                   np.ndarray[double, ndim=1, mode="c"] consts,
                   np.ndarray[double, ndim=1, mode="c"] rates,
                   np.ndarray[double, ndim=1, mode="c"] states,
                   np.ndarray[double, ndim=1, mode="c"] algebraic):
                    
    return computeRates( t,
                       &consts[0],
                       &rates[0],
                       &states[0],
                       &algebraic[0])



###################################
#   FD update
###################################

cdef extern void FDupdate(   double* v_old, # Vm[0:n] IN
                            double* w_old, # rn[0:7*n] IN
                            double* CONSTANTS, # model_constants IN
                            double VOI, # actual time IN
                            double dt,      # IN
                            int n,          # IN
                            int nstates,    # IN
                            int nalg,       # IN
                            double* gamma,   # IN
                            double source,  # current source
                            int rev,        # flag for reversibility
                            double* v_new, # OUT
                            double* w_new
                )

                
def c_FDupdate(  np.ndarray[double, ndim=1, mode="c"] v_old,
                np.ndarray[double, ndim=2, mode="c"] w_old,
                np.ndarray[double, ndim=1, mode="c"] CONSTANTS,
                double VOI,
                double dt,
                int n,
                int nstates,    # IN
                int nalg,       # IN
                np.ndarray[double, ndim=1, mode="c"] gamma, # IN
                double source,
                int rev,
                np.ndarray[double, ndim=1, mode="c"] v_new,
                np.ndarray[double, ndim=2, mode="c"] w_new):
                    
    return FDupdate( &v_old[0],
                    &w_old[0,0],
                    &CONSTANTS[0],
                    VOI,
                    dt,
                    n,
                    nstates,
                    nalg,
                    &gamma[0],
                    source,
                    rev,
                    &v_new[0],
                    &w_new[0,0])
    

###################################
#   Conductance update - Voltage
###################################
  
cdef extern void NOupdate(  double* v, # Vm[0:n] IN
                            long int* gappos, # 
                            double* Cxparams, # 
                            double* gamparams, # 
                            int ndiv, # actual time IN
                            int ngap,
                            int tcond,
                            double* gjnorm,
                            double* gamma
                )

                
def c_NOupdate(  np.ndarray[double, ndim=1, mode="c"] v,
                np.ndarray[long int, ndim=1, mode="c"] gappos,
                np.ndarray[double, ndim=1, mode="c"] Cxparams,
                np.ndarray[double, ndim=1, mode="c"] gamparams,
                int ndiv,
                int ngap,
                int tcond,
                np.ndarray[double, ndim=1, mode="c"] gjnorm,
                np.ndarray[double, ndim=1, mode="c"] gamma):
                    
    return NOupdate( &v[0],
                    &gappos[0],
                    &Cxparams[0],
                    &gamparams[0],
                    ndiv,
                    ngap,
                    tcond,
                    &gjnorm[0],
                    &gamma[0])
    
    
###################################
#   Conductance update - Dynamic
###################################
    
cdef extern void NODupdate(  double* v, # Vm[0:n] IN
                            long int* gappos, # 
                            double* dynparams, # 
                            double* Cxparams, # 
                            double* gamparams, # 
                            int ndiv, # actual time IN
                            int ngap,
                            double* gjnorm,
                            double* gss,
                            double* taug,
                            double* gamma
                )

                
def c_NODupdate(  np.ndarray[double, ndim=1, mode="c"] v,
                np.ndarray[long int, ndim=1, mode="c"] gappos,
                np.ndarray[double, ndim=1, mode="c"] dynparams,
                np.ndarray[double, ndim=1, mode="c"] Cxparams,
                np.ndarray[double, ndim=1, mode="c"] gamparams,
                int ndiv,
                int ngap,
                np.ndarray[double, ndim=1, mode="c"] gjnorm,
                np.ndarray[double, ndim=1, mode="c"] gss,
                np.ndarray[double, ndim=1, mode="c"] taug,
                np.ndarray[double, ndim=1, mode="c"] gamma):
                    
    return NODupdate( &v[0],
                    &gappos[0],
                    &dynparams[0],
                    &Cxparams[0],
                    &gamparams[0],
                    ndiv,
                    ngap,
                    &gjnorm[0],
                    &gss[0],
                    &taug[0],
                    &gamma[0])
    



