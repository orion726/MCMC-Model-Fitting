"""
Run a metropolis-hastings MCMC model fitting procedure on VLA 33 GHz data
of L1251. The modelling approach picks random trial parameters which are then
passed into RADMC (dullemond et al. 2012) to calculate the raditave transfer
equations of a given disk/star setup. The trial disk model is then convolved
with the syntethic beam of the VLA to simulate observations. The model and
33 GHz data are then used to calculated a modifed chi^2 statistic. A given 
trial model will be either accepted or rejected based on comparing the likelihood
function to a random number.

The analysis options include: 

	PDF - calculating the posterior distributon functions (PDF)
	& plotting the walkers of the MCMC
	best_fit - generate a RADMC model of the most probable values
	plot_model_analytics - plot a 2D slice of the density map of the best fit
		model along with a radial profile of Toomre's Q parameter in the
		disk's midplane 
	plot_residual - make a data/model/residual plot of the best fit model

This script also depends on the the mcmc_functions.py script to load in all of the 
functions necessary for the MCMC and analysis.

To run this script you must have:
		
	dirty_imagejuly13.fits - A data FITS file
	pb_imagejuly13.fits - A primary beam FITS file
	beam_imagejuly13.fits - A synthetic beam FITS file
	dustkappa_silicate.inp- The input dust opacity and scattering file 

	The following Python packages in Python 3.6:
	Numpy v1.6.2 or later
	SciPy 0.11.0 or later
	matplotlib 1.2.0 or later
	AstroPy v0.3 or later or Pyfits v3.0.7
	corner - https://doi.org/10.21105/joss.00024
	The input dust kappa file dustkappa_silicate.inp

	The following "standard" python packages:
	math
	os
	times
	sys
	glob

	And the complied RADMC3D binaries - 
	https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_rmcpy/download.html


Analytic plots include:
	PDFs
	walkers
	2D slice of RADMC model
	Radial profile of Toomre's Q parameter
	Data/Model/Residuals


"""

import mcmc_functions as mcmc
import os
import numpy as np

###
# Set which MCMC steps/analysis to run
###

run_mcmc = False
PDF = False
best_fit = False
plot_model_analytics = True
plot_residual = False

###
# Run the MCMC and input the priors, number of chains, length of chains, and 
# the number of threads for the RADMC computation.
###

if run_mcmc == True:

	###
	# Set prior distributions
	# parameter = [Gaussian standard deviation, lower bound, upper bound]
	# These are then packed into a 6x3 array
	###

	mdisk_p = [0.08, -3.0, -0.01]
	R_char_p = [25, 75, 350]
	surf_den_exp_p = [0.1, -2.0, 2.0]
	scale_height_p = [0.08, 0.0, 1.0]
	rstar_p = [0.15, 0.5, 10.0]
	flare_exp_p = [0.025, 0.0, 1.0]

	priors = [mdisk_p, R_char_p, surf_den_exp_p, scale_height_p, rstar_p, flare_exp_p]
	priors = np.reshape(priors, (6,3))

	###
	# Set the number of chains (walkers), links (steps), and threads
	###

	chains = 3
	chain_length = 3
	threads = 8	

	mcmc.run_mcmc_fit(priors, chains, chain_length, threads)


###
# Generate PDFs and a walker plot of the mcmc run
###

if PDF == True:

	import glob

	###
	# Set the input parameters
	###

	filename = "MCMC_100"
	thin = 3
	burn_in = 100

	directory = os.getcwd()
	files = directory+"/results_mcmc/MCMC_chain*"
	names = glob.glob(files)
	os.chdir(directory+"/results_mcmc")

	best_fit_values = mcmc.make_PDF(names, filename, burn_in, thin)

	os.chdir(directory)


###
# Run RADMC using the most probable values from the mcmc
###

if best_fit == True:

	###
	# Move to new working directory
	###

	try:
		os.makedirs("best_fit")
	except:
		sys.exit("Best fit data already exists")
	
	os.system("cp dustkappa_silicate.inp best_fit/")
	folder = os.getcwd()
	os.chdir(folder+"/best_fit")

	###
	# Set the input parameters. Run with PDF = True to get best_fit_values as an array
	###

	params = best_fit_values
	params = [-1.1110504041312177, 263.26507530033655, -0.7389114758823174, 10**(-0.8961546099380229), 3.997329897830427, 10**(-0.6032172713658803)]

	inc = 83.0
	PA = -108.0
	threads = 3
	filename = "best_fit"
	
	mcmc.run_radmc(params, inc, PA, filename, threads)
	os.chdir(folder)

	
###
# Use the best fit RADMC model to creat a 2D slice of the density profile with temperature
# contours and a radial profile of Toomre's Q parameter in the disk midplane. 
###

if plot_model_analytics == True:

	###
	# Move to working directory of best fit model
	# This can only be run if the /best_fit folder exists
	###

	folder = os.getcwd()
	os.chdir(folder+"/best_fit")
	filename = "best_fit"

	mcmc.best_fit_analytics(filename)

	os.chdir(folder)

	
###
# Plot the data/model/residuals of the best fit model.
###

if plot_residual == True:

	###
	# Move to working directory of best fit model
	###

	folder = os.getcwd()
	os.chdir(folder+"/best_fit")
	filename = "residuals"

	mcmc.plot_residuals(filename)

	os.chdir(folder)





