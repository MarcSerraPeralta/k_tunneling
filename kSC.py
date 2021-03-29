#!/usr/bin/env python
import argparse
import sys, os
import numpy as np

from tunneling import *


###########################################################################################

def read_input(f_name):
	"""
	Given an input file in the current working directory returns: 
	list of barriers, T, E_reactants, E_TS, E_products, imaginary frequency of TS, variation of free Gibbs energy from reactants to TS

	The units must be K, Hartree and cm^-1 for Temperature, Energy and Frequency respectively. 

	For more information see advanced help (keyword '--help_adv'). 
	"""
	# Read file
	f = open(f_name, "r")
	data = f.read()
	data = data.split("\n")
	f.close()

	# Process information
	list_barriers = data[0].replace(",", "").split(" ") # deletes all ',' from possible cells
	input_values = [[j for j in line.split(",")] for line in data[1:] if line != ""]
	for i in range(len(input_values)):
		for j in range(len(input_values[i])):
			if input_values[i][j] == "": input_values[i][j] = "0" # solves empty cells
			try:
				input_values[i][j] = float(input_values[i][j])
			except:
				print("ERROR: Could not convert an input value to float: {}".format(input_values[i][j]))
				sys.exit(0)

	# Checks
	DEFAULT_BARRIERS = ["classical", "squared", "parabolic", "eckart", "eckart-approx", "wigner"]
	list_barriers = [i for i in list_barriers if i != ""]
	for b in list_barriers:
		if b not in DEFAULT_BARRIERS: 
			print("ERROR: '{}' is not a valid barrier. The default barriers are: ".format(b) + ", ".join(DEFAULT_BARRIERS))
			sys.exit(0)

	for i in range(len(input_values)):
		if len(input_values[i]) != 7: 
			print("ERROR: Values do not follow the required 7-column CSV format:\n{}".format([line for line in data[1:] if line != ""][i]))
			sys.exit(0)

	# Prepare input data
	input_values = np.array(input_values)
	T, E_r, E_TS, E_p, freq_TS, G_r, G_TS = input_values[:,0], input_values[:,1], input_values[:,2], input_values[:,3], input_values[:,4], input_values[:,5], input_values[:,6]
	AG_TS = G_TS - G_r

	# Change Hartrees to eV (1 Hartree = 27.211386245988 eV, https://en.wikipedia.org/wiki/Hartree)
	factor = 27.211386245988 # ev * Hartree^-1
	E_r, E_TS, E_p, AG_TS = E_r*factor, E_TS*factor, E_p*factor, AG_TS*factor

	return list_barriers, T, E_r, E_TS, E_p, freq_TS, AG_TS


def gaussian2input(f_reactant, f_TS, f_product, f_output):
	# REACTANT
	f = open(f_reactant, "r")
	data = {}
	for line in f:
		if "Temperature " in line:
			T = float(line.replace("Temperature ", "").split("Kelvin")[0]) #  Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.
			T = "{:0.2f}".format(T)
			data[T] = []
		if "Sum of electronic and zero-point Energies=" in line:
			data[T] += [float(line.replace("Sum of electronic and zero-point Energies=", ""))] #  Sum of electronic and zero-point Energies=          -3046.798852
		if "Sum of electronic and thermal Free Energies=" in line:
			data[T] += [float(line.replace(" Sum of electronic and thermal Free Energies=", ""))] #   Sum of electronic and thermal Free Energies=        -3046.839950
	f.close()
	# TRANSITION STATE
	f = open(f_TS, "r")
	freq_TS = 0 # will only select the first frequency
	for line in f:
		if ("Frequencies --" in line) and (freq_TS == 0):
			freq_TS = float(line.split("Frequencies --")[1].split("            ")[0]) # Frequencies --  -1272.9599                86.8910               109.8062
			if freq_TS > 0:
				print("WARNING: No imaginary frequency found in TS")
		if "Temperature " in line:
			T = float(line.replace("Temperature ", "").split("Kelvin")[0]) 
			T = "{:0.2f}".format(T)
			if T not in data: print("ERROR: No T={} (from TS) found in Reactants".format(T)); sys.exit(0)
			data[T] += [freq_TS]
		if "Sum of electronic and zero-point Energies=" in line:
			data[T] += [float(line.replace("Sum of electronic and zero-point Energies=", ""))]
		if "Sum of electronic and thermal Free Energies=" in line:
			data[T] += [float(line.replace(" Sum of electronic and thermal Free Energies=", ""))]
	f.close()
	# PRODUCT
	f = open(f_product, "r")
	for line in f:
		if "Temperature " in line:
			T = float(line.replace("Temperature ", "").split("Kelvin")[0]) 
			T = "{:0.2f}".format(T)
			if T not in data: print("ERROR: No T={} (from Products) found in Reactants".format(T)); sys.exit(0)
		if "Sum of electronic and zero-point Energies=" in line:
			data[T] += [float(line.replace("Sum of electronic and zero-point Energies=", ""))]
	f.close()

	# PREPARE INPUT
	f = open(f_output, "w")
	f.write("classical squared parabolic eckart eckart-approx wigner\n")
	for T in data.keys():
		E_r, G_r, freq_TS, E_TS, G_TS, E_p = data[T]
		# T, E_reactants, E_TS, E_products, imaginary frequency of TS, G_reactants, G_TS
		f.write("{},{},{},{},{},{},{}\n".format(T, E_r, E_TS, E_p, freq_TS, G_r, G_TS))
	f.close()
	return


def general_k(barrier, T, E_r, E_TS, E_p, freq_TS, AG_TS, PRINT=False):
	"""
	Returns rate constant from the specified values. 
	The energies must be in eV, temperatures in K and frequencies in cm^-1. 
	"""
	k = 0
	if "classical" == barrier:
		k = k_classical(T, AG_TS)
	if "squared" == barrier:
		k = kSC_squared(T, [E_r, E_TS, E_p, freq_TS, AG_TS], PRINT=PRINT)
	if "parabolic" == barrier:
		k = kSC_parabolic(T, [E_r, E_TS, E_p, freq_TS, AG_TS])
	if "eckart" == barrier:
		k = kSC_eckart(T, [E_r, E_TS, E_p, freq_TS, AG_TS], PRINT=PRINT)
	if "eckart-approx" == barrier:
		k = kSC_eckart_approx(T, [E_r, E_TS, E_p, freq_TS, AG_TS], PRINT=PRINT)
	if "wigner" == barrier:
		k = kSC_wigner(T, [E_r, E_TS, E_p, freq_TS, AG_TS], PRINT=PRINT)

	return k

def general_coeff(barrier, T, E_r, E_TS, E_p, freq_TS, PRINT=False):
	"""
	Returns tunneling coefficient from the specified values. 
	The energies must be in eV, temperatures in K and frequencies in cm^-1. 
	"""
	kappa = 0
	if "classical" == barrier:
		kappa = coeff_classical(T)
	if "squared" == barrier:
		kappa = coeff_squared(T, [E_r, E_TS, E_p, freq_TS], PRINT=PRINT)
	if "parabolic" == barrier:
		kappa = coeff_parabolic(T, [E_r, E_TS, E_p, freq_TS])
	if "eckart" == barrier:
		kappa = coeff_eckart(T, [E_r, E_TS, E_p, freq_TS], PRINT=PRINT)
	if "eckart-approx" == barrier:
		kappa = coeff_eckart_approx(T, [E_r, E_TS, E_p, freq_TS], PRINT=PRINT)
	if "wigner" == barrier:
		kappa = coeff_wigner(T, [E_r, E_TS, E_p, freq_TS], PRINT=PRINT)

	return kappa


def general_prob(barrier, E, E_r, E_TS, E_p, freq_TS):
	"""
	Returns tunneling probability from the specified values. 
	The energies must be in eV and frequencies in cm^-1. 
	"""
	prob = 0
	if "classical" == barrier:
		prob = prob_classical(E, E_TS)
	if "squared" == barrier:
		prob = prob_squared(E, [E_r, E_TS, E_p, freq_TS])
	if "eckart" == barrier:
		prob = coeff_eckart(E, [E_r, E_TS, E_p, freq_TS])

	return prob

###########################################################################################

help_info = \
"""
========================
RATE CONSTANT CALCULATOR
========================

Script that calculates the semi-classical rate constant, tunneling coefficient and tunneling probability for a given reaction. 
All the barriers included are described in 'TYPES OF BARRIERS'. 

INPUT FILE
=============
The input file must be in the same directory as this script is executed. 
It is a txt or csv file with the following format:
* First line: names of the type of barriers to calculate separated by spaces (see 'TYPES OF BARRIERS' for the options)
* Next lines: values of T, E_reactants, E_TS, E_products, imaginary frequency of TS, G_reactants, G_TS in the CSV format

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
    Energy in [Hartree]
    Free Gibbs Energy in [Hartree]
    Frequency in [cm^-1]
Output:
    Rate constant in [s^-1]
    Tunneling coefficient does not have units
    Tunneling probability does not have units

TYPES OF BARRIERS
=============
Description of the included barriers:
    classical       : no tunelling considered
    squared         : squared barrier
    parabolic       : parabolic barrier
    eckart          : Eckart barrier
    eckart-approx   : Eckart barrier with high barrier approximation
    wigner          : Eckart barrier with high barrier and high temperature approximation 

For further information read each function's description in the repository:
https://github.com/MarcSerraPeralta/k_tunneling

"""

input_file = \
"""%nprocs=8
%chk=/users/....chk
%mem=20Gb
#p
freq=noraman temperature=
B3LYP/6-311G(d,p)
guess=read geom=checkpoint
 
NAME
 
 0  1

"""

###########################################################################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--help_adv', action='store_true', help="Full explanation of the script")
	parser.add_argument('-n', type=str, default=None, help="Name of the input file")
	parser.add_argument('-o', type=str, default=None, help="Name of the output file")
	parser.add_argument('-log', action='store_true', help="Saves data as: ln(k) and 1/T")
	parser.add_argument('-verbose', action='store_true', help="Prints additional information of the calculation")
	parser.add_argument('-Tx', action='store_true', help="Prints crossover temperature")
	parser.add_argument('-gaussian', type=str, nargs='+', default=None, help="Name of the input gaussian files (reactants, TS and products)")
	parser.add_argument('-gaussian_T', type=str, default=None, help="Name of the input gaussian files for multiple themochemical calculations")
	parser.add_argument('-T', type=float, nargs='+', default=None, help="With -gaussian_T, specifies Tmin, Tmax and DeltaT")
	parser.add_argument('-prob', action='store_true', help="Calculates tunneling probability")
	parser.add_argument('-coeff', action='store_true', help="Calculates tunneling coefficient")

	args = parser.parse_args()

	# NO ARGS PARSED
	if (args.n is None) and (not args.Tx) and (args.gaussian is None) and (args.gaussian_T is None) and (not args.prob) and (not args.coeff):
		print(help_info); sys.exit(0)

	# HELP
	if args.help_adv: print(help_info); sys.exit(0)

	# CORSSOVER TEMPERATURE
	if args.Tx:
		# check
		if args.n is None: print("ERROR: Specify input file (-n)"); sys.exit(0)
		if args.n not in os.listdir(): print("ERROR: '{}' not in working directory".format(args.n)); sys.exit(0)

		list_barriers, T, E_r, E_TS, E_p, freq_TS, AG_TS = read_input(args.n)

		for i in range(len(T)):
			print("Tx = {0:0.2f}".format(Tx(E_TS[i] - E_r[i], freq_TS[i])))
		sys.exit(0)

	# CREATE INPUT CSV FROM GAUSSIAN OUTPUT FILES
	if args.gaussian is not None:
		# check
		if len(args.gaussian) != 3: print("ERROR: Specify 3 output gaussian files (reactants, TS and products) (-gaussian)"); sys.exit(0)
		for name in args.gaussian:
			if name not in os.listdir(): print("ERROR: '{}' not in working directory".format(name)); sys.exit(0)
		if args.o is None: print("ERROR: Specify name of the input file for kSC.py (-o)"); sys.exit(0)

		gaussian2input(*args.gaussian, args.o)
		sys.exit(0)

	# CREATE GAUSSIAN INPUT FILE FOR MULTIPLE GAUSSIAN THERMOCHEMICAL CALCULATIONS
	if args.gaussian_T is not None:
		# check
		if len(args.T) != 3: print("ERROR: Specify 3 values in -T (Tmin, Tmax, DeltaT)"); sys.exit(0)
		if (args.T[1] - args.T[0])*args.T[2] <= 0: print("ERROR: Step increment and Tmin and Tmax are not compatible"); sys.exit(0)

		Ti, Tf, AT = args.T
		list_T = np.arange(Ti, Tf + AT, AT)
		f_output = open(args.gaussian_T, "w")
		f_output.write(input_file.replace("temperature=", "temperature={}".format(list_T[0])) + "\n\n")
		for T in list_T[1:]:
			f_output.write("--link1--\n" + input_file.replace("temperature=", "temperature={}".format(T)) + "\n\n")

		f_output.close()
		sys.exit(0)

	# PERFORM CALCULATIONS
	# checks
	if args.n is None: print("ERROR: Specify input file (-n)"); sys.exit(0)
	if args.n not in os.listdir(): print("ERROR: Input file not in working directory"); sys.exit(0)
	if args.o is None: args.o = (args.n).replace(".csv", "_output.csv")
	if ".csv" not in args.o: print("ERROR: Input file must have CSV extension"); sys.exit(0)

	# PROBABILITY CALCULATION
	if args.prob:
		possible_barriers = ["classical", "squared", "eckart"]
		list_barriers, E, E_r, E_TS, E_p, freq_TS, AG_TS = read_input(args.n)
		list_barriers = [b for b in list_barriers if b in possible_barriers] # deletes not possible barriers

		# change units of E from Hartree to eV as all the others are in eV (1 Hartree = 27.211386245988 eV, https://en.wikipedia.org/wiki/Hartree)
		factor = 27.211386245988 # ev * Hartree^-1
		E = E*factor

		f_output = open(args.o, "w")

		if not args.log: # save data as prob vs E
			f_output.write(",".join(["E"] + ["prob " + i for i in list_barriers]) + "\n")
			for i in range(len(E)):
				f_output.write("{},".format(E[i]))
				prob_all = []
				for barrier in list_barriers:
					prob_all += [general_prob(barrier, E[i], E_r[i], E_TS[i], E_p[i], freq_TS[i])]
				f_output.write(",".join(["{:0.9e}".format(k) for k in prob_all]))
				f_output.write("\n")

		if args.log: # save data as ln(k) vs E
			f_output.write(",".join(["E"] + ["ln(prob) " + i for i in list_barriers]) + "\n")
			for i in range(len(E)):
				f_output.write("{},".format(E[i]))
				prob_all = []
				for barrier in list_barriers:
					prob_all += [general_prob(barrier, E[i], E_r[i], E_TS[i], E_p[i], freq_TS[i])]
				f_output.write(",".join(["{:0.9e}".format(np.log(k)) for k in prob_all]))
				f_output.write("\n")

		f_output.close()
		sys.exit(0)

	# COEFFICIENT CALCULATION
	if args.coeff:
		possible_barriers = ["classical", "squared", "parabolic", "eckart", "eckart-approx", "wigner"]
		list_barriers, T, E_r, E_TS, E_p, freq_TS, AG_TS = read_input(args.n)
		list_barriers = [b for b in list_barriers if b in possible_barriers] # deletes not possible barriers

		f_output = open(args.o, "w")

		if not args.log: # save data as coeff vs T
			f_output.write(",".join(["T"] + ["coeff " + i for i in list_barriers]) + "\n")
			for i in range(len(T)):
				f_output.write("{},".format(T[i]))
				coeff_all = []
				for barrier in list_barriers:
					coeff_all += [general_coeff(barrier, T[i], E_r[i], E_TS[i], E_p[i], freq_TS[i], PRINT=args.verbose)]
				f_output.write(",".join(["{:0.9e}".format(k) for k in coeff_all]))
				f_output.write("\n")

		if args.log: # save data as ln(k) vs 1/T
			f_output.write(",".join(["1/T"] + ["ln(coeff) " + i for i in list_barriers]) + "\n")
			for i in range(len(T)):
				f_output.write("{},".format(1/T[i]))
				coeff_all = []
				for barrier in list_barriers:
					coeff_all += [general_coeff(barrier, T[i], E_r[i], E_TS[i], E_p[i], freq_TS[i], PRINT=args.verbose)]
				f_output.write(",".join(["{:0.9e}".format(np.log(k)) for k in coeff_all]))
				f_output.write("\n")

		f_output.close()
		sys.exit(0)
	

	# RATE CONSTANT
	possible_barriers = ["classical", "squared", "parabolic", "eckart", "eckart-approx", "wigner"]
	list_barriers, T, E_r, E_TS, E_p, freq_TS, AG_TS = read_input(args.n)
	list_barriers = [b for b in list_barriers if b in possible_barriers] # deletes not possible barriers

	f_output = open(args.o, "w")

	if not args.log: # save data as k vs T
		f_output.write(",".join(["T"] + ["k " + i for i in list_barriers]) + "\n")
		for i in range(len(T)):
			f_output.write("{},".format(T[i]))
			k_all = []
			for barrier in list_barriers:
				k_all += [general_k(barrier, T[i], E_r[i], E_TS[i], E_p[i], freq_TS[i], AG_TS[i], PRINT=args.verbose)]
			f_output.write(",".join(["{:0.9e}".format(k) for k in k_all]))
			f_output.write("\n")

	if args.log: # save data as ln(k) vs 1/T
		f_output.write(",".join(["1/T"] + ["ln(k) " + i for i in list_barriers]) + "\n")
		for i in range(len(T)):
			f_output.write("{},".format(1/T[i]))
			k_all = []
			for barrier in list_barriers:
				k_all += [general_k(barrier, T[i], E_r[i], E_TS[i], E_p[i], freq_TS[i], AG_TS[i], PRINT=args.verbose)]
			f_output.write(",".join(["{:0.9e}".format(np.log(k)) for k in k_all]))
			f_output.write("\n")

	f_output.close()