import cython
import numpy as np
cimport numpy as np


        
###################################
#   Conductance update - Dynamic
###################################
    
cdef extern void update(  double* un,        # Vm_n[0:n] IN
                          double* w,         # w[0:7n] IN and OUT
                          double* CONSTANTS, # constants IN
                          double VOI,       # voi[0:n] IN
                          double dt,         # dt IN
                          int ndofs,      # ndofs IN
                          double* fa         # fa[0:n] IN
                )

                
def c_update(  np.ndarray[double, ndim=1, mode="c"] un,
                np.ndarray[double, ndim=1, mode="c"] w,
                np.ndarray[double, ndim=1, mode="c"] CONSTANTS,
                double VOI,
                double dt,
                int ndofs,
                np.ndarray[double, ndim=1, mode="c"] fa):
                    
    return update( &un[0],
                    &w[0],
                    &CONSTANTS[0],
                    VOI,
                    dt,
                    ndofs,
                    &fa[0])
    
    
###################################
#   Effective conduction update
###################################
    

cdef extern void effcond(double* gradu,     # IN gradu[0:n] 
                         double* y_arr,     # IN
                         double* s_arr,     # IN
                         double* y_range,   # IN
                         double* s_range,   # IN
                         int npoints,       # IN
                         double* sig        # OUT
                )
                
def c_effcond(  np.ndarray[double, ndim=1, mode="c"] gradu,
                   np.ndarray[double, ndim=1, mode="c"] y_arr,
                   np.ndarray[double, ndim=1, mode="c"] s_arr,
                   np.ndarray[double, ndim=1, mode="c"] y_range,
                   np.ndarray[double, ndim=1, mode="c"] s_range,
                   int npoints,
                   np.ndarray[double, ndim=1, mode="c"] sig):
                    
    return effcond( &gradu[0],
                       &y_arr[0],
                       &s_arr[0],
                       &y_range[0],
                       &s_range[0],
                       npoints,
                       &sig[0])

    
    

###################################
#   EP model Functions
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



