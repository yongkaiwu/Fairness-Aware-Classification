# solve libGL.so.1 error
import matplotlib
matplotlib.use ("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from fair_classification.functions import *

linestyles = {
	'solid': (0, ()),
	'loosely dotted': (0, (1, 10)),
	'dotted': (0, (1, 5)),
	'densely dotted': (0, (1, 1)),
	'loosely dashed': (0, (5, 10)),
	'dashed': (0, (5, 5)),
	'densely dashed': (0, (5, 1)),
	'loosely dashdotted': (0, (3, 10, 1, 10)),
	'dashdotted': (0, (3, 5, 1, 5)),
	'densely dashdotted': (0, (3, 1, 1, 1)),
	'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
	'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
	'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}


def exp2():
	###############
	# adult
	###############
	# parameters
	plt.figure (1)
	plt.style.use ('seaborn-colorblind')
	params = {'text.latex.preamble': [r'\usepackage{amsmath, amssymb}']}
	plt.rcParams.update (params)

	# Zafar-1 line, identity
	df = pd.read_csv ('outputs/adult-Zafar_identity.txt', header=None)
	df.columns = ['tau', 'risk', 'rd']
	plt.plot (df['rd'], df['risk'], color='b', linestyle='-', label='Zafar-1')

	# Zafar-2 line: hinge
	df = pd.read_csv ('outputs/adult-Zafar_hinge.txt', header=None)
	df.columns = ['tau', 'risk', 'rd']
	plt.plot (df['rd'], df['risk'], color='g', linestyle='--', label='Zafar-2')

	# our method
	df = pd.read_csv ('outputs/adult-hinge.txt', header=None)
	df.columns = ['tau', 'risk', 'rd']
	plt.plot (df['rd'], df['risk'], color='r', linestyle=':', label='Our Method')

	# unconstrained
	df = pd.read_csv ('outputs/adult-unconstrained.txt', header=None)
	plt.scatter (df.iloc[0, 1], df.iloc[0, 0], marker='x', s=50, label="Unconstrained")

	# figure setting
	plt.xlim ([-0.10, 0.4])
	plt.ylim ([0.18, 0.31])
	plt.xlabel (r'Risk Difference', fontsize=20)
	plt.ylabel (r'Empirical Loss', fontsize=20)
	plt.gca ().legend (loc='upper right', shadow=True, fontsize='x-large')
	plt.savefig ('outputs/exp2-1.png', bbox_inches='tight')

	###############
	# dutch
	###############
	plt.figure (2)
	plt.style.use ('seaborn-colorblind')
	# Zafar-1 line, identity
	df = pd.read_csv ('outputs/dutch-Zafar_identity.txt', header=None)
	df.columns = ['tau', 'risk', 'rd']
	plt.plot (df['rd'], df['risk'], color='b', linestyle='-', label='Zafar-1')

	# Zafar-2 line: hinge
	df = pd.read_csv ('outputs/dutch-Zafar_hinge.txt', header=None)
	df.columns = ['tau', 'risk', 'rd']
	plt.plot (df['rd'], df['risk'], color='g', linestyle='--', label='Zafar-2')

	# our method
	df = pd.read_csv ('outputs/dutch-hinge.txt', header=None)
	df.columns = ['tau', 'risk', 'rd']
	plt.plot (df['rd'], df['risk'], color='r', linestyle=':', label='Our Method')

	# unconstrained
	df = pd.read_csv ('outputs/dutch-unconstrained.txt', header=None)
	plt.scatter (df.iloc[0, 1], df.iloc[0, 0], marker='x', s=50, label="Unconstrained")

	# figure setting
	plt.xlim ([-0.10, 0.2])
	plt.ylim ([0.18, 0.31])
	plt.xlabel (r'Risk Difference', fontsize=20)
	plt.ylabel (r'Empirical Loss', fontsize=20)
	plt.gca ().legend (loc='upper right', shadow=True, fontsize='x-large')
	plt.savefig ('outputs/exp2-2.png', bbox_inches='tight')


def plot_surrogate():
	# required latex
	x = np.linspace (-2, 2, 1000)

	plt.figure (10, figsize=(7, 4.5))
	plt.style.use ('seaborn-colorblind')
	plt.rc ('text', usetex=True)

	# plt.rc('font', family='serif')
	plt.xlim ([-2, 2])
	plt.ylim ([-1.5, 2.5])

	vfunc = np.vectorize (kappa_zero_one)
	y = vfunc (x)
	line1, = plt.plot (x, y, color='black', linestyle=linestyles['solid'], label='0-1')

	# hinge
	vfunc = np.vectorize (kappa_hinge)
	y = vfunc (x)
	line2, = plt.plot (x, y, color='blue',
					   linestyle=linestyles['dotted'], label=r'$\kappa$=hinge')

	vfunc = np.vectorize (delta_hinge)
	y = vfunc (x)
	line3, = plt.plot (x, y, color='lightblue',
					   linestyle=linestyles['densely dotted'], label=r'$\delta$=hinge')

	# logistic
	vfunc = np.vectorize (kappa_logistic)
	y = vfunc (x)
	line4, = plt.plot (x, y, color='red',
					   linestyle=linestyles['dashed'], label=r'$\kappa$=logistic')

	vfunc = np.vectorize (delta_logistic)
	y = vfunc (x)
	line5, = plt.plot (x, y, color='crimson',
					   linestyle=linestyles['densely dashed'], label=r'$\delta$=logistic')

	# square
	vfunc = np.vectorize (kappa_squared)
	y = vfunc (x)
	line6, = plt.plot (x, y, color='green',
					   linestyle=linestyles['dashdotted'], label=r'$\kappa$=square')

	vfunc = np.vectorize (delta_squared)
	y = vfunc (x)
	line7, = plt.plot (x, y, color='yellowgreen',
					   linestyle=linestyles['densely dashdotted'], label=r'$\delta$=square')

	# plt.gca().legend(loc='lower right', shadow=True, fontsize=10)

	legend1 = plt.legend (
		handles=[line1, line2, line3, line4], loc=0, prop={'size': 15})
	plt.legend (handles=[line5, line6, line7], loc=4, prop={'size': 15})
	plt.gca ().add_artist (legend1)

	try:
		plt.savefig ('outputs/surrogate.png', bbox_inches='tight')
	except PermissionError:
		print ('LaTex is required. Due to inaccessibility, the text rendering fails. ')


if __name__ == '__main__':
	exp2 ()
	plot_surrogate ()
