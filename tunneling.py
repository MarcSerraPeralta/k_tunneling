#!/usr/bin/env python
import numpy as np
import scipy.integrate as integrate
import argparse


"""
========================
RATE CONSTANT CALCULATOR
========================

INPUT ENERGY
=============
Energies must include the Zero Point Energy:
	E(input) = E + ZPE
This applies to the energy and free Gibbs energy of: reactants, transition state, products.
All energies must have the same origin of energies.

UNITS
=============
Input:
	Temperature in [K]
	Energy in [eV]
	Frequency in [cm^-1]
	Free Gibbs Energy in [eV]
Output:
	Rate constant in [s^-1]
	Tunneling coefficient does not have units
	Tunneling probability does not have units

TYPES OF BARRIERS
=============
Description of the included barriers:
	classical 		: no tunelling considered
	squared 		: squared barrier
	parabolic 		: parabolic barrier
	eckart 			: Eckart barrier
	eckart approx 	: Eckart barrier with high barrier approximation
	wigner 			: Eckart barrier with high barrier and high temperature approximation 

For further information read each function's description in the repository:
https://github.com/MarcSerraPeralta/k_tunneling

"""

###########################################################################################

# PROBABILITY OF TUNNEL BARRIERS

def prob_classical(E, E_TS):
	"""
	Returns the classical transmission probability. 
	'E_TS' =  energy of the transition state.

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	if E <= E_TS: 
		return 0
	else: 
		return 1

	return


def prob_squared(E, args):
	"""
	Returns the transmission probability of a squared barrier given an energy E and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS]
	All energies must have same origin of energy. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS = args 		# eV, eV, eV, cm^-1
	E_0 = max([E_r, E_p])				# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Calculate parameters
	Vmax = E_TS - E_r
	A = 8*np.sqrt(Vmax)/hv(freq_TS)

	# Calculate probability
	if E <= E_0:
		prob = 0
	if (E_0 < E) and (E <= E_TS):
		prob = 1/(1 + np.exp(A*np.sqrt(E_TS - E)))
	if (E_TS < E) and (E <= 2*E_TS - E_0):
		prob = 1 - 1/(1 + np.exp(A*np.sqrt(E - E_TS)))
	if E > 2*E_TS - E_0:
		prob = 1

	return prob 	# dimensionless


def prob_eckart(E, args):
	"""
	Returns the transmission probability of an Eckart barrier given an energy E and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS]
	All energies must have same origin of energy. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS = args 		# eV, eV, eV, cm^-1
	freq_TS = np.abs(freq_TS)			# positive frequency
	Vmax = E_TS - E_r 					# eV
	AV = E_p - E_r 						# eV

	# Change origin of energies to the E_reactants (because V_Eckart(x --> -infty) = 0)
	E_TS = E_TS - E_r
	E_p = E_p - E_r
	E = E - E_r
	E_r = 0

	# impossible values of E for tunneling
	if (E < 0) or (E - AV < 0): 
		return 0

	# Eckart parameters
	a = 4*np.pi*np.sqrt(E     ) / ( hv(freq_TS)*( 1/np.sqrt(Vmax) + 1/np.sqrt(Vmax - AV) ) )	# dimensionless
	b = 4*np.pi*np.sqrt(E - AV) / ( hv(freq_TS)*( 1/np.sqrt(Vmax) + 1/np.sqrt(Vmax - AV) ) )	# dimensionless
	d = 2*np.pi*np.sqrt( 4*Vmax*(Vmax - AV)/hv(freq_TS)**2  -  0.25) 				    		# dimensionless

	# Tunnelling transmisison probability
	prob = (np.cosh(a + b) - np.cosh(a - b)) / (np.cosh(a + b) + np.cosh(d))					# dimensionless

	return prob 	# dimensionless


###########################################################################################

# TUNNELLING COEFFICIENTS

def coeff_classical(T):
	"""
	Returns the classical transmission coefficient. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	K_T = 1 	# dimensionless

	return K_T	# dimensionless


def coeff_general(T, prob_function, E_TS, E_0, prob_args=None, PRINT=False):
	"""
	Returns tunnelling coefficient from a tunneling probability given by 'prob_function'. 
	'E_TS' = energy of the transition state. 
	'prob_args' = arguments needed for 'prob_function' (not including the energy, 'E'). 
	'E_0' = max(E_reactives, E_products)).
	'PRINT' = option to print the numerical error of the integral. 

	//!\\ WARNING! 
	This function may came to overflows if the temperature is very small or the energy barrier is bery high. 
	To calculate the rate constant, use 'kSC_general' to minimize this error. 

	The formulas used are from:
	(1) Meana-Pañeda, Rubén & Fernández-Ramos, Antonio. (2010). 
	Tunneling Transmission Coefficients: Toward More Accurate and Practical Implementations. 
	10.1007/978-90-481-3034-4_18. 
	(2) Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	# Change origin of energies to E_0
	E_TS = E_TS - E_0
	E_0 = 0

	# Calculate maximum energy to integrate numerically (P(E_MAX) = 1 - eps)
	eps = 1E-10
	E_MAX = E_TS
	while prob_function(E_MAX, args=prob_args) < 1 - eps:
		E_MAX = E_MAX + E_TS

	# Numeric integration (E_0 to E_MAX)
	f = lambda E, args=None: prob_function(E, args)*np.exp(-(E-E_TS)/kT(T)) # includes the classical tunneling except the \beta factor
	I_N, error = integrate.quad(f, E_0, E_MAX, args=prob_args)
	I_N, error = I_N/kT(T), error/kT(T) # dimensionless

	if PRINT and (I_N != 0): print("Tunneling numerical integral relative error: {0:0.1e} %".format(error/I_N * 100))

	# Analytical integration (E_MAX to infinity)
	I_A = np.exp((E_TS - E_MAX)/kT(T)) # dimensionless

	K_T = I_N + I_A 	# dimensionless

	return K_T 			# dimensionless


def coeff_squared(T, args, PRINT=False):
	"""
	Returns the transmission coefficient of a squared barrier given a temperature T and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS]
	'PRINT' = option to print the numerical error of the integrals. 

	//!\\ WARNING! 
	This function may came to overflows if the temperature is very small or the energy barrier is bery high. 
	To calculate the rate constant, use 'kSC_squared' to minimize this error. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS = args 		# eV, eV, eV, cm^-1
	E_0 = max([E_r, E_p])				# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Calculate parameters
	Vmax = E_TS - E_r
	A = 8*np.sqrt(Vmax)/hv(freq_TS)

	# Two numerical integrations
	f = lambda E: np.exp((E_TS - E)/kT(T)) / (1 + np.exp(A*np.sqrt(E_TS - E)))
	I_N1, error = integrate.quad(f, E_0, E_TS)
	I_N1, error = I_N1/kT(T), error/kT(T) # dimensionless

	if PRINT and (I_N1 != 0): print("Tunneling numerical integral (1) relative error: {0:0.1e} %".format(error/I_N1 * 100))

	f = lambda E: np.exp((E_TS - E)/kT(T)) / (1 + np.exp(A*np.sqrt(E - E_TS)))
	I_N2, error = integrate.quad(f, E_TS, 2*E_TS - E_0)
	I_N2, error = I_N2/kT(T), error/kT(T) # dimensionless

	if PRINT and (I_N2 != 0): print("Tunneling numerical integral (2) relative error: {0:0.1e} %".format(error/I_N2 * 100))

	# Calculate tunneling coefficient
	K_T = 1 + I_N1 - I_N2 	# dimensionless
	
	return K_T				# dimensionless


def coeff_parabolic(T, args):
	"""
	Returns the transmission coefficient of a parabolic barrier given a temperature T and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS]

	//!\\ WARNING! 
	This function may came to overflows if the temperature is very small or the energy barrier is bery high. 
	To calculate the rate constant, use 'kSC_parabolic' to minimize this error. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS = args 		# eV, eV, eV, cm^-1

	# Parameters
	freq_TS = np.abs(freq_TS)			# positive frequency
	Vmax = E_TS - E_r					# eV
	AV = E_p - E_r 						# eV
	Vb = Vmax - max([0, AV])			# eV
	K = 2*np.pi*kT(T)/hv(freq_TS) 		# dimensionless

	# Tunnelling transmisison probability
	eps = 1E-7 # lower numerical error for 1/(1-K) when K --> 1

	if K > 1 + eps:
		K_T = np.pi / (K*np.sin(np.pi/K)) + np.exp((1 - K)*Vb/kT(T)) / (1 - K)

	if (K <= 1 + eps) and (K >= 1 - eps):
		K_T = Vb / kT(T) 					

	if K < 1 - eps:
		K_T = (np.exp((1 - K)*Vb/kT(T)) - 1)/ (1 - K)

	return K_T	# dimensionless


def coeff_eckart(T, args, PRINT=False):
	"""
	Returns the transmission coefficient of an Eckart barrier given a temperature T and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS]
	'PRINT' = option to print the numerical error of the integral. 

	//!\\ WARNING! 
	This function may came to overflows if the temperature is very small or the energy barrier is bery high. 
	To calculate the rate constant, use 'kSC_eckart' to minimize this error. 

	Read 'prob_eckart' and 'coeff_general' description for further information. 
	"""
	E_r, E_TS, E_p, freq_TS = args 		# eV, eV, eV, cm^-1
	E_0 = max([E_r, E_p])				# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Calculate transmission coefficient
	K_T = coeff_general(T, prob_eckart, E_TS, E_0, prob_args=[E_r, E_TS, E_p, freq_TS], PRINT=PRINT)	# dimensionless

	return K_T 	# dimensionless


def coeff_eckart_approx(T, args, PRINT=False):
	"""
	Returns the transmission coefficient of an Eckart barrier given a temperature T and barrier's energy values ('args'). 
	Uses the approximation of gamma >> 1. 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS]
	'PRINT' = option to print the numerical error of the integral and the 'gamma' value. 

	//!\\ WARNING! 
	This function may came to overflows if the temperature is very small or the energy barrier is bery high. 
	To calculate the rate constant, use 'kSC_eckart_approx' to minimize this error. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS = args 		# eV, eV, eV, cm^-1
	E_0 = max([E_r, E_p])				# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Avoid error in E_p > E_r
	if E_r < 0: E_r = 0

	# Calculate gamma
	Vmax = E_TS - E_r
	gamma = 2*np.pi*Vmax / hv(freq_TS)
	if PRINT and (gamma >= 1): print("gamma: {0:0.2f} (should be >>1)".format(gamma))
	if gamma < 1: print("WARNING: gamma={0:0.2f} (should be >>1)".format(gamma))

	# Calculate maximum 'energy' to integrate numerically (X_MAX) 
	eps = 15 		# error = 10^(-eps) 
	x_MAX = gamma*( (1 + eps*np.log(10)/(2*gamma))**2 - 1 ) 

	# Numeric integration (E_0 to E_MAX)
	A = hv(freq_TS)/(2*np.pi*kT(T)) 
	f = lambda x: np.exp(-A*x)/(1 + np.exp(2*gamma*(1 - np.sqrt(1 + x/gamma))))
	I_N, error = integrate.quad(f, -gamma, x_MAX)
	I_N, error = A*I_N, A*error 		# dimensionless 

	if PRINT and (I_N != 0): print("Tunneling numerical integral relative error: {0:0.1e} %".format(error/I_N * 100))

	# Analytical integration (E_MAX to infinity)
	I_A = np.exp(-A*x_MAX) 				# dimensionless 

	# Calculate the transmission coefficient	
	K_T = I_N + I_A						# dimensionless

	return K_T


def coeff_wigner(T, args, PRINT=True):
	"""
	Returns the transmission coefficient of an Eckart barrier given a temperature T and barrier's energy values ('args'). 
	Uses the approximation of gamma >> 1 and high temperatures (delta >> 1). 
	'args' = [freq_TS] or [E_reactants, E_TS, E_products, imaginary freq_TS] 
	'PRINT' = print checks for the approximations. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	if len(args)==1: # no checks can be performed
		freq_TS = args[0]
		if PRINT: print("No checks of the assumptions can be done (more inputs needed)")
		K_T = 1 + (hv(freq_TS)/kT(T))**2 / 24
		return K_T

	E_r, E_TS, E_p, freq_TS = args 		# eV, eV, eV, cm^-1
	E_0 = max([E_r, E_p])				# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Calculate gamma and delta
	Vmax = E_TS - E_r
	gamma = 2*np.pi*Vmax / hv(freq_TS)
	delta = 2*np.pi*kT(T) / hv(freq_TS)
	if PRINT and (gamma >= 1): print("gamma: {0:0.2f} (should be >>1)".format(gamma))
	if gamma < 1: print("WARNING: gamma={0:0.2f} (should be >>1)".format(gamma))
	if PRINT and (delta >= 1): print("T={1:0.2f} delta: {0:0.2f} (should be >>1)".format(delta, T))
	if delta < 1: print("WARNING: T={1:0.2f} delta={0:0.2f} (should be >>1)".format(delta, T))

	# Calculate the transmission coefficient
	K_T = 1 + (hv(freq_TS)/kT(T))**2 / 24	# dimensionless

	return K_T	# dimensionless


###########################################################################################

# SEMI-CLASSICAL AND CLASSICAL RATE CONSTANTS

def k_classical(T, AG_TS):
	"""
	Returns the classic rate constant using Eyring equation given a temperature T and barrier's free Gibbs energy values ('args').
	'AG_TS' = variation of free Gibbs energy from reactives to the transition state. 
	
	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	k = kT(T) * np.exp(-AG_TS/kT(T))/ hv(1) # cm^-1
	k = k * c()								# s^-1

	return k # s^-1


def kSC_general(T, prob_function, E_TS, E_0, AG_TS, prob_args=None, PRINT=False):
	"""
	Returns rate constant from barrier given by 'prob_function' at temperature T. 
	'E_TS' = the energy of the transition state. 
	'prob_args' = arguments needed for 'prob_function' (not including the energy, 'E'). 
	'E_0' = max(E_reactives, E_products)).
	'AG_TS' = free Gibbs energy variation from the reactants to TS. 


	The formulas used are from:
	(1) Meana-Pañeda, Rubén & Fernández-Ramos, Antonio. (2010). 
	Tunneling Transmission Coefficients: Toward More Accurate and Practical Implementations. 
	10.1007/978-90-481-3034-4_18. 
	(2) Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	# Change origin of energies to E_0
	E_TS = E_TS - E_0
	E_0 = 0

	# Calculate maximum energy to integrate numerically (P(E_MAX) = 1 - eps)
	eps = 1E-10
	E_MAX = E_TS
	while prob_function(E_MAX, args=prob_args) < 1 - eps:
		E_MAX = E_MAX + E_TS

	# Numeric integration (E_0 to E_MAX)
	f = lambda E, args=None: prob_function(E, args)*np.exp(-(E-E_TS + AG_TS)/kT(T)) # includes the classical tunneling except the \beta factor
	I_N, error = integrate.quad(f, E_0, E_MAX, args=prob_args)
	I_N, error = I_N/hv(1), error/hv(1) # cm^-1

	if PRINT and (I_N != 0): print("Tunneling numerical integral relative error: {0:0.1e} %".format(error/I_N * 100))

	# Analytical integration (E_MAX to infinity)
	I_A = np.exp((E_TS - E_MAX)/kT(T))*kT(T)/hv(1) # cm^-1

	k = I_N + I_A 	# cm^-1
	k = k * c()		# s^-1

	return k # s^-1


def kSC_squared(T, args, PRINT=False):
	"""
	Returns the rate constant of a squared barrier given a temperature T and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS, AG_TS]
	'PRINT' = option to print the numerical error of the integrals. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS, AG_TS = args 		# eV, eV, eV, cm^-1, eV
	E_0 = max([E_r, E_p])						# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Calculate parameters
	Vmax = E_TS - E_r
	A = 8*np.sqrt(Vmax)/hv(freq_TS)

	# Two numerical integrations
	f = lambda E: np.exp((E_TS - E - AG_TS)/kT(T)) / (1 + np.exp(A*np.sqrt(E_TS - E)))
	I_N1, error = integrate.quad(f, E_0, E_TS)
	I_N1, error = I_N1/hv(1), error/hv(1) # cm^-1

	if PRINT and (I_N1 != 0): print("Tunneling numerical integral (1) relative error: {0:0.1e} %".format(error/I_N1 * 100))

	f = lambda E: np.exp((E_TS - E - AG_TS)/kT(T)) / (1 + np.exp(A*np.sqrt(E - E_TS)))
	I_N2, error = integrate.quad(f, E_TS, 2*E_TS - E_0)
	I_N2, error = I_N2/hv(1), error/hv(1) # cm^-1

	if PRINT and (I_N2 != 0): print("Tunneling numerical integral (2) relative error: {0:0.1e} %".format(error/I_N2 * 100))

	# Calculate tunneling coefficient
	k = kT(T)*np.exp(-AG_TS/kT(T))/hv(1) + I_N1 - I_N2 	# cm^-1
	k = k * c() # s^-1
	
	return k 	# s^-1


def kSC_parabolic(T, args):
	"""
	Returns the semi-classic rate constant of a parabolic barrier given a temperature T and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS, AG_TS]
	
	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS, AG_TS = args 		# eV, eV, eV, cm^-1, eV

	# Parameters
	freq_TS = np.abs(freq_TS)			# positive frequency
	Vmax = E_TS - E_r					# eV
	AV = E_p - E_r 						# eV
	Vb = Vmax - max([0, AV])			# eV
	K = 2*np.pi*kT(T)/hv(freq_TS) 		# dimensionless

	# Tunnelling transmisison probability
	eps = 1E-7 # lower numerical error for 1/(1-K) when K --> 1

	if K > 1 + eps:
		k = np.pi*np.exp(-AG_TS/kT(T))/ (K*np.sin(np.pi/K)) + np.exp(((1 - K)*Vb - AG_TS)/kT(T)) / (1 - K)

	if (K <= 1 + eps) and (K >= 1 - eps):
		k = Vb*np.exp(-AG_TS/kT(T)) / kT(T) 					

	if K < 1 - eps:
		k = (np.exp( ((1 - K)*Vb - AG_TS)/kT(T) ) - np.exp(-AG_TS/kT(T)))/ (1 - K)

	k = k*kT(T)/hv(1) 	# cm^-1
	k = k * c()			# s^-1

	return k	# s^-1


def kSC_eckart(T, args, PRINT=False):
	"""
	Returns the rate constant of an Eckart barrier given a temperature T and barrier's energy values ('args'). 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS, AG_TS]
	'PRINT' = option to print the numerical error of the integrals. 

	Read 'kSC_general' description for further information. 
	"""
	E_r, E_TS, E_p, freq_TS, AG_TS = args 		# eV, eV, eV, cm^-1, eV
	E_0 = max([E_r, E_p])						# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Calculate rate constant
	k = kSC_general(T, prob_eckart, E_TS, E_0, AG_TS, prob_args=[E_r, E_TS, E_p, freq_TS], PRINT=PRINT)	# s^-1

	return k # s^-1


def kSC_eckart_approx(T, args, PRINT=False):
	"""
	Returns the rate constant of an Eckart barrier given a temperature T and barrier's energy values ('args'). 
	Uses the approximation of gamma >> 1. 
	'args' = [E_reactants, E_TS, E_products, imaginary freq_TS, AG_TS]
	'PRINT' = option to print the numerical error of the integrals and the 'gamma' value. 

	The formulas used are from:
	Bao, J. L., & Truhlar, D. G. (2017). Variational transition state theory: theoretical framework and recent developments. 
	Chemical Society Reviews, 46(24), 7548–7596. 
	doi:10.1039/c7cs00602k 
	"""
	E_r, E_TS, E_p, freq_TS, AG_TS = args 		# eV, eV, eV, cm^-1, eV
	E_0 = max([E_r, E_p])						# eV

	# Change origin of energies to E_0
	E_r = E_r - E_0
	E_TS = E_TS - E_0
	E_p = E_p - E_0
	freq_TS = np.abs(freq_TS)
	E_0 = 0

	# Avoid error in E_p > E_r
	if E_r < 0: E_r = 0

	# Calculate gamma
	Vmax = E_TS - E_r
	gamma = 2*np.pi*Vmax / hv(freq_TS)
	if PRINT and (gamma >= 1): print("gamma: {0:0.2f} (should be >>1)".format(gamma))
	if gamma < 1: print("WARNING: gamma={0:0.2f} (should be >>1)".format(gamma))

	# Calculate maximum 'energy' to integrate numerically (X_MAX) 
	eps = 30 		# error = 10^(-eps) 
	x_MAX = gamma*( (1 + eps*np.log(10)/(2*gamma))**2 - 1 ) 

	# Numeric integration (E_0 to E_MAX)
	A = hv(freq_TS)/(2*np.pi*kT(T)) 
	f = lambda x: np.exp(-A*x - AG_TS/kT(T))/(1 + np.exp(2*gamma*(1 - np.sqrt(1 + x/gamma))))
	I_N, error = integrate.quad(f, -gamma, x_MAX)
	I_N, error = A*kT(T)*I_N/hv(1), A*kT(T)*error/hv(1)	# cm^-1

	if PRINT and (I_N != 0): print("Tunneling numerical integral relative error: {0:0.1e} %".format(error/I_N * 100))

	# Analytical integration (E_MAX to infinity)
	I_A = kT(T)*np.exp(-A*x_MAX - AG_TS/kT(T))/hv(1) 	# cm^-1

	# Calculate the rate constant	
	k = I_N + I_A										# cm^-1
	k = k * c()											# s^-1

	return k # s^-1


def kSC_wigner(T, args, PRINT=False):
	"""
	Returns the transmission coefficient of an Eckart barrier given a temperature T and barrier's energy values ('args'). 
	Uses the approximation of gamma >> 1 and high temperatures (delta >> 1). 
	'args' = [freq_TS, AG_TS] or [E_reactants, E_TS, E_products, imaginary freq_TS, AG_TS] 
	'PRINT' = option to the checks for the approximations. 
	"""
	kappa = coeff_wigner(T, args[:-1], PRINT=PRINT)	# dimensionless
	k_c = k_classical(T, args[-1]) 					# cm^-1

	k_SC = kappa*k_c 								# cm^-1
	k_SC = k_SC * c()								# s^-1

	return k_SC # s^-1


###########################################################################################
###########################################################################################

# TUNNELING CROSSOVER TEMPERATURE

def Tx(AE_TS, freq_TS):
	"""
	Returns de tunneling crossover temperature, which indicates that below of which tunneling is important. 
	'AE_TS' = variation of internal energy (including ZPE) from reactants to TS
	'freq_TS' = imaginary frequency of TS
	
	Reference:
	Fermann, J. T., & Auerbach, S. (2000). Modeling proton mobility in acidic zeolite clusters: II. 
	Room temperature tunneling effects from semiclassical rate theory. 
	The Journal of Chemical Physics, 112(15), 6787–6794. doi:10.1063/1.481318 
	"""
	freq_TS = np.abs(freq_TS)
	T_x = hv(freq_TS)*AE_TS/(kT(1)*(2*np.pi*AE_TS - hv(freq_TS)*np.log(2))) # K
	return T_x # K


# CONSTANTS AND UNITS

def kT(T):
	"""
	Returns:
	kT [eV] = kB[eV·K^-1]*T[K] where 'kB' is the Boltzmann constant

	Input: 
	T [K] temperature
	
	kB = 1380649/16021766340 # eV * K^-1
	Reference: https://en.wikipedia.org/wiki/Boltzmann_constant
	"""
	kB = 1380649/16021766340 # eV * K^-1
	return kB*T # eV


def hv(v):
	"""
	Returns:
	hv [eV] = h[eV·s]*c[cm·s^-1]*v[cm^-1] where 'h' is the Plank's constant and 'c' the speed of light
	
	Input:
	v [cm^-1] frequency

	h*c = 1.23984193 eV * um
	Reference: https://en.wikipedia.org/wiki/Planck_constant
	"""
	hc = 1.23984193E-4 	# eV * cm
	return hc*v 		# eV

def c():
	"""
	Returns:
	c [cm/s] where 'c' is the speed of light
	"""
	c = 29979245800 # cm / s
	return c