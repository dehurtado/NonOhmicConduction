#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
    
def Parameters(model):
    params = {}
    
    # Spatial parameters 
    ncells = 64             
    lcell = 0.1
    params['ncells'] = 64        # -, number of cells
    params['lcell'] = 0.1        # mm, cell length
    params['L'] = ncells*lcell   # mm, cell length
    if model == 'NOM' or model == 'LHM':
        params['dx'] = 0.1          # mm, spatial discretization
        params['dt'] = 1e-3         # ms, temporal discretization
    else:
        params['dx'] = 0.01          # mm, spatial discretization
        params['ndiv'] = 10          # -, division of each cell
        params['dt'] = 5e-5          # ms, temporal discretization
    
    # Time parameters
    params['Tf'] = 3000.         # ms, total simulation time
    
    # Stimulus parameters
    params['period'] = 500.      # ms, period of stimulus
    params['source'] = -35.      # uA/mm2, stimulus magnitude
    params['tdur'] = 2.          # ms, duration time for the stimulus
    
    # Geometric parameters        
    a = 0.011           
    params['a'] = a                   # mm, radius of cell
    params['Acell'] = np.pi*a**2      # mm2, transverse area of cells
    
    # Conductivity parameters
    RCG = 2.
    gjc = 1.
    params['RCG'] = RCG             # -, ratio between capacitive and geometrical areas
    params['Am'] = 2./a*RCG         # mm, surface to area ratio
    params['Cm'] = 0.01             # uF/mm2, membrane capacitance
    params['gjc'] = gjc              # -, gap junction coupling       
    params['rho_myo'] = 1/0.667         # Ohm-m, myoplasm resistance
    params['rd0'] = 0.15            # Ohm-m-mm, Acell/gj0 
    params['eta'] = 1.              # -, modulation parameter for gjnorm
    
    # Electrophyisiology model parameters
    params['nvar'] = 8          # for Luo Rudy, Number of variables for EP Model
    params['nconsts'] = 24      # for Luo Rudy, Number of constants for EP Model
    params['nalg'] = 25         # for Luo Rudy, Number of algebraic for EP Model
    
    # Output parameters
    params['vtkpath']  = './'+ model +'_vtks/'     # folder where to save .vtk files
    params['logpath']  = './'+ model +'_logs/'     # folder where to save logfile
    params['fname'] = model + '_gjc'+str(gjc*100)  # especific file name
    params['dtsave'] = 0.1                              # Time step for saving vtks
    
    # Connexin parameters
    params['Cx'] = 'Cx43-Cx43'  # connexin name
    
    # Time dynamics
    params['sigg'] = 1.48e-3    
    params['kg'] = 0.0493
    
    # Homogenization parameters
    params['delta'] = 1e-4
    params['epsilon'] = lcell
    
    # Model
    if model == 'DM' or model == 'LHM':
        cond_type = 'O'
    elif model == 'DNOM':
        cond_type = 'NO'
    elif model == 'DNODM':
        cond_type = 'NOD'
    elif model == 'NOM':
        cond_type = 'NO'
        
    params['model'] = model
    params['condtype'] = cond_type 
    params['tcond'] = 'ss' 
    params['k'] = 2*params['epsilon']
    
    return params 
    