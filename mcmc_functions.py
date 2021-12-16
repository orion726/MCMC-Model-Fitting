"""
Set of functions required to run a Metropolis-Hastings MCMC model fitting procedure with Gibbs Sampling
and visualize the results. 

The function have the following Python depedenices:
	math
	numpy
	random
	os
	astropy.io
	time
	sys

In addition, to generate a RADMC model, the following dependecies mustbe met:
	Numpy v1.6.2 or later
	SciPy 0.11.0 or later
	matplotlib 1.2.0 or later
	AstroPy v0.3 or later or Pyfits v3.0.7
	Complied RADMC3D binaries - 
	https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_rmcpy/download.html

"""

import math as m
import numpy as np
import random
import os
from astropy.io import fits as pf
import matplotlib.pyplot as plt
import time
import sys


def alpha_calc(chi2_0, chi2_1):
	"""
	Calculate the alpha parameter for the MCMC acceptance/rejection based
	on the current and previous values of the chi squared statistic. 

	alpha is the likelihood function where L \propto e^(0.5*( chi2_i - chi2_(i+1) )

	Input:
		chi2_0, chi2_1 - the chi squared values for the previous and current models
	
	Ouput:
		number in range from 0 to 1 (float, dimensionless)
	"""

	alpha = np.exp(0.5*(chi2_0-chi2_1))

	if alpha > 1.0:
		alpha = 1
	
	return alpha


def best_fit_analytics(filename):
	"""
	This function will make 2 analytical plots of the best fit RADMC model. The
	first plot is a test of the gravitational instability throughout the disk. 
	Toomre's Q parameter is calculated along the disk midplane and plotted.

	The 2nd plot is a 2D slice of the disk density with 100 and 200 K temperature
	contour overlaid. These indicate the upper and lower bounds where the water
	snowline in the disk would be.

	RADMC (Dullemon et al. 2012) is used to generate raditivate transfer model s
	of a circumstellar disk. 
	https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_rmcpy/download.html

	
		In order to call the script, the dependcies for RADMC must be met including:
			Numpy v1.6.2 or later
			SciPy 0.11.0 or later
			matplotlib 1.2.0 or later
			AstroPy v0.3 or later or Pyfits v3.0.7

	Input:
		filename - name for saving the plots
	Ouput:
		filename_Q.png - Q vs radius plot for best fit model
		filename_2D_density_slice.png - 2D desnity slice of the disk with temperature contours
	"""

	import radmc3dPy as r

	###
	# Define some physical constants
	###

	G = 6.67430E-11 #m^3 kg^-1 s^2
	au = 1.496E11 # au in m
	M_star = 1.6 * 1.989E30 # kg
	k_B = 1.38064852E-23 # boltzmann m^2 kg s^-2 K-1 
	mu = 2.4
	m_h = 1.6735575E-27 #hydrogen mass kg

	###
	# Load in the already generated RADMC model
	# Must be in the same directory as the files
	# dust_temperature.bdat, dust_denisty.binp, and dustkappa
	###

	data2 = r.analyze.readData(dtemp=True, binary = True)
	data = r.analyze.readData(ddens=True)

	r_t = data2.grid.x/r.natconst.au
	r_t_m = data.grid.x/100. # convert to m

	###
	# Get the surface density profile of the disk
	####

	vol = data.grid.getCellVolume()
	surf = np.zeros([data.grid.nx, data.grid.nz], dtype=np.float64)
	diff_r2 = (data.grid.xi[1:] ** 2 - data.grid.xi[:-1] ** 2) * 0.5
	diff_phi = data.grid.zi[1:] - data.grid.zi[:-1]
	for ix in range(data.grid.nx):
	        surf[ix, :] = diff_r2[ix] * diff_phi

	mass = np.zeros([data.grid.nx, data.grid.nz], dtype=np.float64)
	sigma = np.zeros([data.grid.nx, data.grid.nz], dtype=np.float64)
	
	mass[:, :] = (vol * data.rhodust[:, :, :, 0]).sum(1)
	sigma[:, :] = mass / surf *(100*100.) * 100. / 1000. #convert cm^2 to m^2, apply gas-to-dust ratio, convert g to kg

	####
	# Get the average temperature profile in the disk midplane
	####

	T_disk = 0
	for i in range(30):
		T_disk += data2.dusttemp[:,42,i,0].T
	T_disk = T_disk/30.

	###
	# Calculate Toomre's Q parameter
	###

	c_s = (k_B * T_disk /(mu*m_h))**0.5
	omega = ((G * M_star)/(r_t_m)**3)**0.5

	Q = c_s * omega / (m.pi * G * sigma[:,0])

	###
	# Plot Q as a function of r in the disk midplane
	###

	plt.style.use('ggplot')

	plt.plot(r_t, Q, color = "blue", label = "L1251 GI")
	plt.plot([0,500], [1.4,1.4], color = "black", label = "Q = 1.4")
	plt.legend()
	plt.yscale("log")
	plt.xscale("log")
	plt.xlabel("Radius (au)")
	plt.ylabel("Q")
	filenameQ = filename+"Q.png"
	plt.savefig(filenameQ)
	plt.show()

	###
	# Plot 2D slice of density with temperature contours overlaid
	###

	c = plt.contourf(data.grid.x/r.natconst.au, np.pi/2.-data.grid.y, np.log10(data.rhodust[:,:,0,0].T), 30)

	r.analyze.plotSlice2D(data2, var='dtemp', plane='xy', ispec=0, log=True, linunit='au', contours=True, clev=[100, 200], clcol='k', cllabel=True)
	cb = plt.colorbar(c)
	plt.xlabel('Radius (au)')
	plt.xlim([0.5, 250])
	plt.ylabel(r'$\pi/2-\theta$')
	plt.xscale('log')
	filename2D = filename+"2D_density_slice.png"
	plt.savefig(filename2D)
	plt.show()


def chi_sq(obs, model, sigma_rms, beam_maj, beam_min, beam_px):
	"""
	Calculate the chi squared for the MCMC.
	
	chi squared is (data-model)^2 / SIGMA^2 summed over all elements of the array

	The SIGMA required here is the sigma_rms of the data multipled by the beam area in pixels
	(see e.g., White et al. 2017 https://doi.org/10.1093/mnras/stw3303).


	Input:
		obs - the observational data as a 2D array
		model - the trial model as a 2D array
		sigma_rms - the rms of the observational data in the same units as 
			the data (e.g. uJy/beam)
		beam_maj - the beam major axis in arcseconds
		beam_min - the beam minor axis in arcseconds
		beam_px - the data cell size in arcseconds
		
	Ouput:
		the Chi Squared statistic (float, dimensionless)

	SIGMA = ((1E-6)*(pi*(3.94 / 2.3548)*(2.06 / 2.3548) / (0.1)^2))^2

	"""
	
	diff = (obs-model)
	SIGMA = ((sigma_rms)*(m.pi*(beam_maj / 2.3548)*(beam_min / 2.3548) / (beam_px)**2))
	chi2 = np.sum((diff*diff) / (SIGMA**2))

	return chi2


def convolve_fft(array, kernel, padding = False, padding_amount = 0):
	"""
	
	Convolve the model with the beam. This function loads in a skymodel as 
	"array" the beam as "kernel". The array and kernerl are padded to reduce
	the edge effects of the final image. The array is Fourier transformed,
	then kernel is shifted and Fourier transformed, the arrays are multipled 
	together, then the real component is returned.

	The array and kernel need to be the same dimensions in order for this 
	function to run. If they are not the same size, additional padding can be
	added. 

	Input:
		array - input 2D array of the trial model
		kernel - the synthetic beam 2D array
		padding	True/False (default True)
		padding amount - amoutn of padding to add to the array (default 0)
	Output:
		2D array of the convolved image (same size and units as the input array)
	
	"""
	###
	# Add Padding
	###

	if padding == True:
		array = np.pad(array, pad_width = padding_amount)
		kernel = np.pad(kernel, pad_width = padding_amount, mode = 'linear_ramp')

	###
	# Do the convolution
	###
	arrayfft = np.fft.fft2(array)
	kernfft = np.fft.fft2(np.fft.ifftshift(kernel))
	fftmult = arrayfft * kernfft
	rifft = np.fft.ifft2(fftmult)
	rrifft = np.real(rifft)
	
	###
	# Trim the resulting file back down to the original size
	###
	
	if padding == True:
		p = padding_amount-1
		a = array.shape[0] - padding_amount - 1
		b = array.shape[1] - padding_amount - 1
		rrifft = rrifft[p:a, p:b]

	return rrifft


def get_data(name):
	"""
	Read in a .txt filename and remove the preceeding strings of text. A 2D array
	of the MCMC chain is returned.

	Input:
		MCMC chain filename
	Output:
		The input array stripped of any string values
	"""
	
	data_file = open(name, 'r')
	data_file_tmp = open('_tmpfile_.txt', 'w')
	for line in data_file:
		data_file_tmp.write(line.strip().split('string ')[1])
		data_file_tmp.write("\n")

	data_file_tmp.close()
	data_file.close()
	Data = np.loadtxt('_tmpfile_.txt')

	return Data


def make_model(params_model, inc, PA, PB, DB, threads = 1):
	"""
	Generate the RADMC model from trial parameters. 
	This function is calls by mcmc_functions.gen_RT() to generate the trial
	RADMC-3D disk model and mcmc_functions.convolve_fft() to make the simulated
	observations.

	The function will use the radmc3dPy functions analyze.writeDefaultParfile('ppdisk') and 
	setup.problemSetupDust() to create the apprortae input files for RADMC. The radmc3d 
	commands mctherm and image will then be made to calculate the dust termperature 
	structure of the disk and compute the ray-tracing of the of the image.

	After a trial disk model is created by mcmc_functions.gen_RT(), the model is 
	multipled by the primary beam file (PB) in order to attenuate the total flux.
	The model is then convolved by the synthetic beam with the function 
	mcmc_functions.convolve_fft().
 
	In order to call the function, the dependcies for RADMC must be met including:
		Numpy v1.6.2 or later
		SciPy 0.11.0 or later
		matplotlib 1.2.0 or later
		AstroPy v0.3 or later or Pyfits v3.0.7
		Complied RADMC3D binaries - 
	https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_rmcpy/download.html

	After the disk structure is generated, 3 point sources are added to the skymodel
	in order to account for the disks of A, B, and C in L1251.

	Input:
		mdiskL - total disk mass (dust+gas) in M_sun
		R_char - characteristic disk radius in au
		surf_den_exp - the surface denisty exponent (dimensionless)
		scale_height - the r/r_char scale height fraction (dimensionless)
		rstar - stellar radius in R_sun
		flare_exp - the flaring exponent (dimensionless)
		inc - disk inclination in degrees
		PA - disk position angle in degrees
		threads - the number of threads for parallel computing in generating 
			the disk model (dimensionless)
		PB - the primary beam 2D array
		DB - the dirty beam 2D array

	Output:
		2D array of the tfull sky model that has been attenuated by the primary
		beam and convolved with the synthetic beam.
	"""


	###
	# Generate RADMC model
	###

	name = "sky_model"
	#params_rad = params_model
	#params_rad[0] = 10**(params_rad[0])
	run_radmc(params_model, inc, PA, name, threads)
	model0 = pf.getdata('sky_model.fits')
	model0 = model0[:,:]

	###
	# Mutliply by the primary beam to attenuate flux
	###
	
	m = model0 * PB
	
	###
	# Convolve with the synthetic beam
	###

	sim = convolve_fft(m, DB)

	return sim


def make_PDF(names, filename, burn_in=100, thin = 3):
	"""
	This function reads in all the MCMC chains in the .../results folder, cacluates the most
	most probable values, generate a walker plot for all parameters, and plots the posterior distribution 
	funcitons for all parameters.

		Requires:
			corner - https://doi.org/10.21105/joss.00024
			glob
			os
			matplotlib.pyplot
			numpy
			
	Input:
		filename - name of the PDF and walker figures
		burn_in - number of links in the chain to remove for burn-in. Default is 100
		thin - amount of thinning to do for the chain. Default is 3
	Ouput:
		best_fit - the most probable values for the free parameters
		filename.png - The posterior distribution functions (PDF)
		filename_walkers.png - The walkers for each prameter. 
	"""

	###
	# Initialize walker plots
	###	

	full_data = np.zeros(6)

	fig = plt.figure()
	plt.style.use('ggplot')

	ax1 = fig.add_subplot(2,3,1)
	ax2 = fig.add_subplot(2,3,2)
	ax3 = fig.add_subplot(2,3,3)
	ax4 = fig.add_subplot(2,3,4)
	ax5 = fig.add_subplot(2,3,5)
	ax6 = fig.add_subplot(2,3,6)

	for i in range(len(names)):
		###
		# Read in data
		###
		data = get_data(names[i])

		###
		# Plot walkers
		###

		ns = np.arange(len(data[:,2]))
		ax1.plot(ns,data[:,1])
		ax2.plot(ns,data[:,2])
		ax3.plot(ns,data[:,3])
		ax4.plot(ns,data[:,4])
		ax5.plot(ns,data[:,5])		
		ax6.plot(ns,data[:,6])

		###
		# Remove burn-in, trim columns, apply thinning
		###

		dels = np.arange(0,burn_in)
		data = np.delete(data, dels, axis=0)
		data = np.delete(data, 0,1)
		data = np.delete(data, 6,1)
		data = data[::thin]


		full_data = np.row_stack((full_data, data))

	filename_walker = filename+"_walker.png"
	plt.savefig(filename_walker)
	plt.show()

	###
	# Clean up arrays
	###

	data = full_data
	data = np.delete(data,0,axis = 0)
	data[:,3] = np.log10(data[:,3]) 
	data[:,5] = np.log10(data[:,5])
	
	###
	# Make histograms for each parameter
	###

	bins=20
	a1, edgea1 = np.histogram(data[:,0], bins=bins, density = True)
	ind = np.argmax(a1)
	l = np.linspace(min(data[:,0]), max(data[:,0]), bins)
	print(l[ind])
	m1 = l[ind]

	a2, edgea2 = np.histogram(data[:,1], bins=bins, density = True)
	ind = np.argmax(a2)
	l2 = np.linspace(min(data[:,1]), max(data[:,1]), bins)
	print(l2[ind])
	m2 = l2[ind]

	a3, edgea3 = np.histogram(data[:,2], bins=bins, density = True)
	ind = np.argmax(a3)
	l3 = np.linspace(min(data[:,2]), max(data[:,2]), bins)
	print(l3[ind])
	m3 = l3[ind]

	a4, edgea4 = np.histogram(data[:,3], bins=bins, density = True)
	ind = np.argmax(a4)
	l4 = np.linspace(min(data[:,3]), max(data[:,3]), bins)
	print(l4[ind])
	m4 = l4[ind]
	
	a5, edgea5 = np.histogram(data[:,4], bins=bins, density = True)
	ind = np.argmax(a5)
	l5 = np.linspace(min(data[:,4]), max(data[:,4]), bins)
	print(l5[ind])
	m5 = l5[ind]

	a6, edgea6 = np.histogram(data[:,5], bins=bins, density = True)
	ind = np.argmax(a6)
	l6 = np.linspace(min(data[:,5]), max(data[:,5]), bins)
	print(l6[ind])
	m6 = l6[ind]

	###
	# Make corner plot
	###

	import corner

	label = [r"$\rm M_{disk}$", r"$\rm R_{c}$", r"$ \gamma$", r"$ \rm h_{c}$", r"$ \rm R_{star}$", r"$\psi$"]

	dat = corner.corner(data, verbose = True,  labels=label, plot_contours = True, bins=20, label_kwargs={"fontsize": 20}, fill_contours=True, ret=True, plot_datapoints=False, truths=[m1,m2,m3,m4,m5,m6])

	filename = filename+"_PDF.png"
	plt.savefig(filename)
	plt.show()

	return [m1, m2, m3, m4, m5, m6]


def plot_residuals(filename):
	"""
	Plot the observational data, the best fit model, and the data - model residuals.

	Input:
		filename - input filename for the figure
	Output:
		filename.png - 3 panel plot of the data/model/residuals
	"""

	from matplotlib.patches import Ellipse

	###
	# Load in the FITS images
	###

	try:
		beam_full = pf.getdata('beam_imagejuly13.fits')
	except:
		os.system("cp ../beam_imagejuly13.fits .")
		beam_full = pf.getdata('beam_imagejuly13.fits')

	beam = beam_full[0,0,:,:]

	try:
		image0 = pf.getdata('best_fit.fits')
	except:
		sys.exit("No best fit image")

	model_image = image0[:,:]

	try:
		obs1 = pf.getdata('dirty_imagejuly13.fits')
	except:
		os.system("cp ../dirty_imagejuly13.fits .")
		obs1 = pf.getdata('dirty_imagejuly13.fits')

	obs = obs1[0,0,:,:]

	###
	# Make a simulated image
	###

	sim_obs = convolve_fft(model_image, beam, padding = True, padding_amount = 125)

	###
	# Make the residual map
	###

	diff = obs - sim_obs

	###
	# Set some parameters for the plotting
	###

	length = 25
	beam_min = 2.0
	beam_maj = 3.0
	beam_PA = -65

	##
	# Image1 - Data
	##

	plt.style.use('ggplot')

	fig = plt.figure()
	ax1 = fig.add_subplot(1,3,1, aspect='equal')

	lev = np.linspace(np.min(obs), np.max(sim_obs), num=200)
	dat1 = plt.contourf(obs,origin = 'lower', levels = lev, cmap='RdBu_r', extent = [-length/2., length/2., -length/2., length/2.])
	plt.xlabel(r'$\Delta \alpha$ ["]', fontsize = 10)
	plt.ylabel(r'$\Delta \delta$ ["]', fontsize = 10)
	el = Ellipse((-9.3,-9.3), beam_min, beam_maj, angle = beam_PA, color = 'black', hatch='//////', fill = False)
	ax1.add_patch(el)
	plt.title(r"Observations")
	
	##
	# Image2 - Model
	##
	
	ax2 = fig.add_subplot(1,3,2, aspect='equal')

	lev = np.linspace(np.min(obs), np.max(sim_obs), num=200)
	dat2 = plt.contourf(sim_obs,origin = 'lower', levels = lev, cmap='RdBu_r', extent = [-length/2., length/2., -length/2., length/2.])
	plt.xlabel(r'$\Delta \alpha$ ["]', fontsize = 10)
	ax2.yaxis.set_visible(False)
	el = Ellipse((-9.3,-9.3), beam_min, beam_maj, angle = beam_PA, color = 'black', hatch='//////', fill = False)
	ax2.add_patch(el)
	plt.title(r"Model")

	##
	# Image3 - Residuals
	##

	ax3 = fig.add_subplot(1,3,3, aspect='equal')

	lev = np.linspace(np.min(diff), np.max(obs), num=200)
	dat3 = plt.contourf(diff,origin = 'lower', levels = lev, cmap='RdBu_r', extent = [-length/2., length/2., -length/2., length/2.])
	ax3.yaxis.set_visible(False)
	plt.xlabel(r'$\Delta \alpha$ ["]', fontsize = 10)
	el = Ellipse((-9.3,-9.3), beam_min, beam_maj, angle = beam_PA, color = 'black', hatch='//////', fill = False)
	ax3.add_patch(el)
	plt.title(r"Residuals")

	###
	# Add Colorbar
	###

	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.8, 0.25, 0.02, 0.49])
	cbar = fig.colorbar(dat1, cax=cbar_ax, ticks=[-.000045, 0, .000045, .000100, .000200, .000300, .000400, .000500])
	cbar.set_ticklabels([ "-45", "0.0", "45", "100", "200", "300", "400", "500"])
	cbar.set_label(r'$\rm \mu Jy  \, beam^{-1}$', labelpad = 1, fontsize = 10)

	###
	# Adjust Plot and save
	###

	plt.subplots_adjust(wspace=0)
	fig.set_size_inches(10,4.5)
	fig.subplots_adjust(wspace=0, hspace=0)
	filename+=".png"
	plt.savefig(filename)
	plt.show()


 
def rand_vars(params0, priors):
	"""
	Generate a trial set of parameters for the MCMC. 2 of the input parameters will
	be randomly selected. New values will be selected from a Gaussian distribution 
	centered on the previous value and with a width passed in. If the value falls 
	outside of the prior range, a new value will be selected.
	
	Input:
		params0 - An array consisting of the 6 previous trial parameters.
		priors - A 6x3 array with each row representing the width of the 
			Gaussian, the lower bound, and the upper bound.
		
	Output:
		trial_values - An array of 6 newly selected trial values.	

	"""
	
	###
	# Randomly pick 2 parameters to vary
	###

	a=0
	b=0

	while a==b:
		a = round(random.random()*5)+1
		b = round(random.random()*5)+1

	###
	# Generate new trial parameters within prior ranges
	###

	redo = 0

	while redo < 2:
		redo = 0
	
		
		if a == 1.0 or b == 1.0:
			mdisk_scale = random_normal(params0[0], priors[0,0]);
			if priors[0,1] < mdisk_scale < priors[0,2]: 
				redo += 1
			
		else:
			mdisk_scale = params0[0]
		

		if a == 2.0 or b == 2.0:
			R_char_scale = random_normal(params0[1], priors[1,0]);
			if priors[1,1] < R_char_scale < priors[1,2]: 
				redo += 1
			
		else:
			R_char_scale = params0[1]
		

		if a == 3.0 or b == 3.0:
			surf_den_exp_scale = random_normal(params0[2], priors[2,0]);
			if priors[2,1] < surf_den_exp_scale < priors[2,2]: 
				redo += 1
			
		else:
			surf_den_exp_scale = params0[2]
		

		if a == 4.0 or b == 4.0:
			scale_height_scale = random_normal(params0[3], priors[3,0]);
			if priors[3,1] < scale_height_scale < priors[3,2]:
				redo += 1
			
		else:
			scale_height_scale = params0[3]	
		

		if a == 5.0 or b == 5.0:
			rstar_scale = random_normal(params0[4], priors[4,0]);
			if priors[4,1] < rstar_scale < priors[4,2]:
				redo += 1
			
		else:
			rstar_scale = params0[4]	
		

		if a == 6.0 or b == 6.0:
			flare_exp_scale = random_normal(params0[5], priors[5,0]);
			if priors[5,1] < flare_exp_scale < priors[5,2]:
				redo += 1
			
		else:
			flare_exp_scale = params0[5]	
		
	
	trial_values =  [mdisk_scale, R_char_scale, surf_den_exp_scale, scale_height_scale, rstar_scale, flare_exp_scale]
	
	return trial_values


def random_normal(mu, sig):
	"""
	Random normal distribution.

	Input:
		mu - the mean value of the distribution
		sig - the standard deviation of the distribution

		
	Ouput:
		Random number from normal distibution (float)

	"""

	r = m.sqrt( -2.0*np.log10(random.random()) )
	theta = 2.0*m.pi*random.random()
	mu += sig*r*m.sin(theta)

	return mu


def run_mcmc_fit(priors, chains, chain_length, threads):
	"""
	Run a Metropolis-Hastings MCMC algorithm with Gibbs sampling to find the most 
	probable model	parameters of the L1251 circumstellar disk. The model is compared 
	to the VLA 33 GHz observations of the L1251 system which has 4 circumstellar disks.
	A trial model is only generated for L1251 D and point-point source fluxes are 
	added to represent the other 3 disks for the purposes of image convolution
	and chi^2 calculation.

	Trial model parameters are generated with rand_vars() by randomly selecting 2 
	parameters to update. The new values for are selected from a Gaussian 
	distibution centeredon the previous parameter value and with a width and prior 
	range indiciated by the priors input.

	A given set of trial parameters is then used to generate a trial disk model in
	RADMC (see below) by calling make_model() which calls run_radmc(). The model is 
	multipled by the primary beam file (PB) in order to attenuate the total flux.
	The model is then convolved by the synthetic beam with the function convolve_fft().

	A chi squared value is calculated with chi_sq() by comparing the current model to
	the observational data. 

	The alpha value for the MCMC is then calculated with alpha_calc(). alpha is the 
	likelihood function where L \propto e^(0.5*( chi2_i - chi2_(i+1) ). If alpha > 1
	then alpha == 1.0
	
	The given model is then accpeted if a random number from a uniform distribution [0,1] 
	is larger than alpha. If the model is accepted it is recorded on the given chain. 
	If the model is rejected, the previous model parameters are recorded. 

	The mcmc runs for chains x chains_length times. Each chain is daved in a separate
	.txt file under the subfolder results_mcmc. The total acceptance % for a given 
	chain and the run time are recorded in acceptance.txt.

	
 
	In order to call the RADMC-related functions, the dependcies for RADMC must 
	be met including:
		Numpy v1.6.2 or later
		SciPy 0.11.0 or later
		matplotlib 1.2.0 or later
		AstroPy v0.3 or later or Pyfits v3.0.7
		Complied RADMC3D binaries - 
	https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_rmcpy/download.html


	Input:
		priors - A 6x3 array with each row representing the width of the 
			Gaussian, the lower bound, and the upper bound.
		chains - integer number of chains to be ran
		chain_length - integer number of links in each chain

	Output:
		MCMC_chain_* - txt files for each MCMC chain
		accapetance.txt - list of acceptance % and run time for each chain
	"""

	import time

	###
	# Check that the appropriate run environment is set up
	###

	current_dir = os.getcwd()

	try:
		os.makedirs("results_mcmc")
	except:
		sys.exit("A results_mcmc directory already exists. Consider renaming or deleting it.")


	if os.path.isfile("dustkappa_silicate.inp") == False:
		sys.exit("Dust Kappa file not found (required for RADMC)")	

	try:
		import radmc3dPy
	except:
		sys.exit("RADMC3D Python environment not found")

	
	###
	# Load in the data, synthetic beam, and primary beam
	###

	try:
		image_full = pf.getdata('dirty_imagejuly13.fits')
		obs_data = image_full[0,0,:,:]
	except:
		sys.exit("Data file not found")

	try:
		image_full = pf.getdata('beam_imagejuly13.fits')
		syn_beam = image_full[0,0,:,:]
	except:
		sys.exit("Synthetic beam file not found")

	try:
		image_full = pf.getdata('pb_imagejuly13.fits')
		primary_beam = image_full[0,0,:,:]
	except:
		sys.exit("Primary beam file not found")

	try:
		os.chdir(current_dir+"/mcmc_working")
	except:
		os.makedirs("mcmc_working")
		os.system("cp dustkappa_silicate.inp mcmc_working/")
		os.chdir(current_dir+"/mcmc_working")
		

	###
	# Make the file to record acceptance % and run times
	###

	outfile2 = current_dir+"/results_mcmc/acceptance.txt"
	output_accept = open(outfile2,"a")

	###
	# Run the MCMC for the number of iteration specified by chains
	###

	for i in range(chains):

		###
		# Define some starting values
		###

		chi2_start = 1000000000
		m = i+1 
		outfile = current_dir+"/results_mcmc/MCMC_chain_%s.txt"%(m)
		output = open(outfile,"w")
		accept = 0
		index = 0
		start_i = time.perf_counter()

		###
		# Initialize the random starting points
		###

		mdisk = -1*(0.75 + 0.5*random.random())  # Disk Mass [M_sun]
		R_char =  225 + 75*random.random() # Disk characteristic radius [au]
		surf_den_exp = -1.0 + random.random()   # Surface density exponent 
		scale_height = 0.2*random.random()   # Pressure scale height at r/r_char
		rstar = 2.0 + 4.0*random.random()  # Stellar radius [R_sun]
		flare_exp = 0.05 + 0.2*random.random()	# Degree of flaring

		###
		# Fixed Parameters
		###

		inclination = 53 # disk inclination [degrees]
		position_angle = -108.0 # Disk positions angle East of North [degrees]
		sigma_rms = 1E-6 # uJy/beam rms from the dirty image
		beam_maj = 3.94 # beam major axis in arcseconds
		beam_min = 2.06 # beam minor axis in arcseconds
		beam_px = 0.1 # cell size in arcseconds

		params = [mdisk, R_char, surf_den_exp, scale_height, rstar, flare_exp]


		###
		# Run the MCMC chain for chain_length steps
		###

		while index < chain_length:

			###
			# Draw random variables
			###

			params_i = rand_vars(params, priors)

			###
			# Make the trial model	
			###

			model =  make_model(params_i, inclination, position_angle, primary_beam, syn_beam, threads)

			###
			# Calculate the chi squared
			###

			chi2_tmp = chi_sq(model, obs_data, sigma_rms, beam_maj, beam_min, beam_px);


			###
			# Accept/Reject the trial model
			###

			if random.random() <= alpha_calc(chi2_start,chi2_tmp):

				###
				# Record results
				###

				string = "string %s %s %s %s %s %s %s %s \n"%(index, params_i[0],  params_i[1],  params_i[2],  params_i[3],  params_i[4],  params_i[5], chi2_tmp) 
				output.write(string)
				
				###
				# Update Parameters	
				###

				params = params_i
				chi2_start = chi2_tmp		
				accept+=1

			else:
				###
				# Record results
				###

				string = "string %s %s %s %s %s %s %s %s \n"%(index, params[0],  params[1],  params[2],  params[3],  params[4],  params[5], chi2_start) 
				output.write(string)
		
			index +=1

	

		output.close()

		###
		# Record acceptance % and time
		###
		accept_total = accept/chain_length
		time_step = time.perf_counter() - start_i
		string = "%s %s %s \n"%(m, accept_total, time_step)
		output_accept.write(string)

	output_accept.close()
	os.chdir(current_dir)



def run_radmc(model_parameters, inc, PA, model_filename = "sky_model", threads = 1):
	"""
	Run the RADMC command (Dullemon et al. 2012) to generate a raditivate transfer model 
	of the circumstellar disk. 
	https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_rmcpy/download.html

	
		In order to call the script, the dependcies for RADMC must be met including:
			Numpy v1.6.2 or later
			SciPy 0.11.0 or later
			matplotlib 1.2.0 or later
			AstroPy v0.3 or later or Pyfits v3.0.7

	Input:
		model_parameters - mdisk, R_char, surf_den_exp, scale_height, rstar, flare_exp
		model_filename - name of the FITS file
		threads - number of threads for radmc calculation (defualt 1)
		
		
	Output:
		sky_model.fits - the RADMC generated disk model plus the flux values for the other sources 
			in the field of view of the observations.
	"""

	import radmc3dPy as r

	###
	# Fixed Parameters of L1251 D needed for RADMC
	###

	RA_DEC                    = '22h35m23.46s +75d17m07.6s' # RA and DEC of target
	d                         = 350    # distance in pc
	T_star                    = 10000  # stellar effective temp in K
	npix                      = 250    # output image size [npix x npix] pixels
	sizeau                    = 8750    # output image size in au
	sigma_type                = 1      # Surface density type (0 - polynomial, 1 - exponential outer edge 	
	nphot			  = int(1e5) # number of photons in the calculation
	modified_random_walk      = 1 # 0 or 1, set to 1 for more efficient calculation
	gsmax                     = 15000.0  # Maximum grain size
	gsmin                     = 0.1  # Minimum grain size
	wbound                    = [0.1, 7.0, 25., 1.5e4]  # Boundraries for the wavelength grid
	ngs = 1
	q = -3.5

	###
	# Free Disk Parameters
	###

	mdisk = 10**(model_parameters[0])
	R_char = model_parameters[1] 
	surf_den_exp = model_parameters[2] 
	scale_height = model_parameters[3] 
	rstar = model_parameters[4] 
	flare_exp = model_parameters[5]

	###
	# Set default parameter file
	###

	r.analyze.writeDefaultParfile('ppdisk') 

	###
	# Add the input disk parameters
	###

	r.setup.problemSetupDust('ppdisk', xbound = ['0.5*au','50*au', '500.0*au'], tstar = '[%s]'%T_star, rstar = '[%s*rs]'%rstar, mdisk='%s*ms'%mdisk, sigma_type =sigma_type, plh=flare_exp, hrpivot = '%s*au'%R_char, plsig1 = surf_den_exp, hrdisk= scale_height, rdisk = '%s*au'%R_char, gsmax = gsmax, modified_random_walk=modified_random_walk, nphot = nphot)

	###
	# Run the radmc disk calculation
	###

	os.system("radmc3d mctherm setthreads %s"%(threads))


	###
	# Remove intermidate files and run the ray-tracing
	###


	try:
		os.system("rm image.out")
	except:
		print("no image to delete")
	
	try:
		os.system("rm sky_model.fits")
	except:
		print("no image to delete")

	os.system("radmc3d image setthreads %s lambda 9080 incl %s posang %s npix %s sizeau %s"%(threads, inc,PA, npix, sizeau))


	im2 = r.image.readImage()

	try:
		os.system("rm trial_model.fits")
	except:
		print("no trial model to delete")

	im2.writeFits('trial_model.fits', dpc=d, coord = RA_DEC)

	###
	# Add point sources to the trial mode to account for the 
	# A, B, and C disk components. The fluxes are and locations
	# are taken from the observed values in the dirty image. 
	###

	A_peak = [107,161]
	BC_peak = [139,97]
	D_peak = [161,125]

	image0 = pf.getdata('trial_model.fits')
	model_image = image0[0,:,:]

	model_image = np.roll(model_image, 37, axis = 1) #shift right
	model_image = np.roll(model_image, -1, axis = 0) # shift up
	
	model_image[95,137] = 1.5E-4
	model_image[99,141] = 1.5E-4
	model_image[161,107] = 5.2E-4

	###
	#Save the fits image
	###

	hdu = pf.PrimaryHDU(model_image)
	model_filename+=".fits"
	hdu.writeto(model_filename)

	os.system("rm *.out")



