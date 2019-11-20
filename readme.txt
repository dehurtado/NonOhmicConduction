In this folder are the codes for the simulations made in the paper 
"Non-ohmic tissue conduction in cardiac electrophysiology: upscaling the 
non-linear voltage-dependent conductance of gap junctions"

The folder CModels contains the codes for the cellular models (discrete models).
The folder HModels contains the codes for the homogenized models (continuum
models).

Each folder contains 6 files:
- XM_master.py: Main file, contains the respective codes for each model. 
  An example is also added here.
- Parameters.py: Contains a function that returns a dictionary with all the
  parameters used in the simulations.
- cond_classes.py: Contains the classes for computing the non-ohmic conductivity.
- NOHfunctions.c: Functions written in C to perform the updates. 
- epmodel.c: the EP model writen in C.
- setup.py, bridge.pyx, bridge.c: Needed files to link python with C using cython.

Both model used cython to run. Before running you need to compile the C files
with this command:
	
- python setup.py build_ext --inplace
