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


def general_k(barrier, T, E_r, E_TS, E_p, freq_TS, AG_TS):
	"""
	Returns rate constant from the specified values. 
	The energies must be in eV. 
	"""
	k = 0
	if "classical" == barrier:
		k = k_classical(T, AG_TS)
	if "squared" == barrier:
		k = kSC_squared(T, [E_r, E_TS, E_p, freq_TS, AG_TS])
	if "parabolic" == barrier:
		k = kSC_parabolic(T, [E_r, E_TS, E_p, freq_TS, AG_TS])
	if "eckart" == barrier:
		k = kSC_eckart(T, [E_r, E_TS, E_p, freq_TS, AG_TS])
	if "eckart-approx" == barrier:
		k = kSC_eckart_approx(T, [E_r, E_TS, E_p, freq_TS, AG_TS])
	if "wigner" == barrier:
		k = kSC_wigner(T, [E_r, E_TS, E_p, freq_TS, AG_TS])

	return k

###########################################################################################

help_info = \
"""
========================
RATE CONSTANT CALCULATOR
========================

Script that calculates the semi-classical reaction rate constant for a given reaction. 
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
	classical 		: no tunelling considered
	squared 		: squared barrier
	parabolic 		: parabolic barrier
	eckart 			: Eckart barrier
	eckart-approx 	: Eckart barrier with high barrier approximation
	wigner 			: Eckart barrier with high barrier and high temperature approximation 

For further information read each function's description in the repository:
https://github.com/MarcSerraPeralta/k_tunneling

"""

###########################################################################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--help_adv', action='store_true', help="Full explanation of the script")
	parser.add_argument('-n', type=str, default=None, help="Input file")
	parser.add_argument('-o', type=str, default=None, help="Output file")
	args = parser.parse_args()

	# HELP
	if args.help_adv: print(help_info)

	# CHECK
	if args.n is None: print("ERROR: Specify input file (-n)"); sys.exit(0)
	if args.n not in os.listdir(): print("ERROR: Input file not in working directory"); sys.exit(0)
	if args.o is None: args.o = (args.n).replace(".csv", "_output.csv").replace(".txt", "_output.txt")

	# GET INPUT DATA
	list_barriers, T, E_r, E_TS, E_p, freq_TS, AG_TS = read_input(args.n)

	f_output = open(args.o, "w")
	f_output.write(",".join(["T"] + list_barriers) + "\n")

	for i in range(len(T)):
		f_output.write("{},".format(T[i]))
		k_all = []
		for barrier in list_barriers:
			k_all += [general_k(barrier, T[i], E_r[i], E_TS[i], E_p[i], freq_TS[i], AG_TS[i])]
		f_output.write(",".join(["{:0.9e}".format(k) for k in k_all]))
		f_output.write("\n")

	f_output.close()