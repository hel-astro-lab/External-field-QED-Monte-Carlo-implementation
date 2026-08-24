This is a supplementary code for our paper "Efficient simulation of first-order QED processes in strong electromagnetic fields".

The files "Implement_Synch.py" "Implement_BW.py" contain all the functions needed in the simulations of the Synchrotron and Breit-Wheeler processes. These include the numerically calculated
spectra, auxiliary functions, and cumulative probabilities. Then, their approximations are given, as well as the inverse functions of the cumulative probability approximations, which are used
to generate the energies of the produced particles in the simulations. The files "SynchrotronFigures.py" and "BreitWheelerFigures.py" contain the codes that were used to generate the figures
in the paper.


The folder "Tables" contains dataframes that are relevant especially for producing the figures. First, they contain the values of the optimal auxiliary parameters that were obtained,
by fitting the approximations to the numerical functions and from which the approximating forms of them were derived. There are also dataframes for certain precalculated values of the
numerical functions - using them instead of calculating the values from scratch speeds up the creation of the figures. The details of these dataframes are provided in the folder.
