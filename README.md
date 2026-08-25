This is a supplementary code repository for the paper "Efficient simulation of first-order QED processes in strong electromagnetic fields" by V. Sarjomaa & J. Nättilä (2026)

The files `Implement_Synch.py` `Implement_BW.py` contain all the functions needed in the simulations of the Synchrotron and Breit-Wheeler processes. These include the numerically calculated spectra, auxiliary functions, and cumulative probabilities. Then, their approximations are given, as well as the inverse functions of the cumulative probability approximations, which are used to generate the energies of the produced particles in the simulations. The files `SynchrotronFigures.py` and `BreitWheelerFigures.py` contain the codes that were used to generate the figures in the paper.

The folder `Tables/` contains data frames that are relevant especially for producing the figures. First, they contain the values of the optimal auxiliary parameters that were obtained, by fitting the approximations to the numerical functions and from which the approximating forms of them were derived. There are also data frames for certain pre-calculated values of the numerical functions --- using them instead of calculating the values from scratch speeds up the creation of the figures. The details of these data frames are provided in the folder in `Tables/INSTRUCTIONS`.

Development of the MC algorithms are supported by an ERC grant (ILLUMINATOR, 101114623).
<img align="center" src="https://cdn.jsdelivr.net/gh/natj/natj.github.io@master/images/erc_logo.png">
