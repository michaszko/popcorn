import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats
import scipy.optimize
from math import sqrt, log
from glob import glob

def derivative(df):
    '''
        Calculating derivative of given series of data
    '''
    return pd.Series(np.gradient(df.values.flatten()), df.index, name='slope')

def filter(df):
    '''  
        Resolution of phone sometimes casuing date to oscilate.
        To properly analize data one has to get rid of those
        oscialtions  using some low pass filter.
    '''
    return df.rolling(window=1, center=True).mean()

class biNorm:

	fitted_params = [1, -45, 1, 1, -40, 1]

	def __init__(self, df, i, initial = fitted_params):
		# Histogramizing data
		self.count, self.division = np.histogram(df.values, bins=100)

		# Calculating bins center point
		self.division_ = self.division
		self.division = (self.division[:-1] + self.division[1:]) / 2

		# For plotting
		self.xx = np.linspace(np.min(self.division), np.max(self.division), 1000)

		# Index setting
		self.i = i

		# Initial values for fittng
		self.initial = initial 

	def bi_norm(self, x, *args):
		'''
			Bi normal function
		'''
		k1, m1, s1, k2, m2, s2 = args
		ret = k1 * scipy.stats.norm.pdf(x, loc=m1, scale=s1)
		ret += k2 * scipy.stats.norm.pdf(x, loc=m2, scale=s2)
		return ret
	
	def fit(self):
		'''
			Fitting binormal function to data
		'''
		self.fitted_params, _ = scipy.optimize.curve_fit(self.bi_norm,
			                                            self.division,
			                                            self.count,
			                                            p0=self.initial)
			
	##################
	# Add chi2 and DOF
	##################


	def cut(self):
	    '''
	        Calulating intersection point of binormal distribution
	    '''
	    k1, m1, s1, k2, m2, s2 = self.fitted_params
	    a = s2 * s2 - s1 * s1
	    b = 2 * s1 * s1 * m2 - 2 * s2 * s2 * m1
	    c = s2 * s2 * m1 * m1 - s1 * s1 * m2 * m2 - 2 * \
	        s2 * s2 * s1 * s1 * log(k1 * s2 / (k2 * s1))
	    x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)
	    x2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a)
	    return max(x1,x2)

	def norm(self):
		'''
		Normalizing data by moving means of distributions. Device was not
		calibrated
		'''
		self.noise = min(self.fitted_params[1], self.fitted_params[4])
		self.division  -= self.noise
		self.division_ -= self.noise

		self.fitted_params[1] -= self.noise
		self.fitted_params[4] -= self.noise

		self.xx -= self.noise

	def plot_hist(self):
		'''
		Plot histogram of data
		'''
		plt.hist(self.division_[:-1], self.division_, weights=self.count, label='Data')

	def plot_fit_sum(self):
		'''
		Plot binormal fit (together)
		'''
		plt.plot(self.xx,
         self.bi_norm(self.xx, *self.fitted_params),
         label="I: %.1f  G(%.1f, %.1f) \nII: %.1f  G(%.1f, %.1f)" %
         tuple(self.fitted_params))

	def plot_fit_sep(self):	
		'''
		Plot binormal fit (separately)
		'''	
		plt.plot(self.xx,
		         self.fitted_params[0] *
		         scipy.stats.norm.pdf(self.xx,
		                              loc=self.fitted_params[1],
		                              scale=self.fitted_params[2]),
		         label="%.1f  G(%.1f, %.1f)" % tuple(self.fitted_params[0:3]))
		plt.plot(self.xx,
		         self.fitted_params[3] *
		         scipy.stats.norm.pdf(self.xx,
		                              loc=self.fitted_params[4],
		                              scale=self.fitted_params[5]),
		         label="%.1f  G(%.1f, %.1f)" % tuple(self.fitted_params[3:6]))

	def plot_derivative(self):
		'''
		Plot derivative of fit
		'''
		dx = xx[1]-xx[0]
		y = self.bi_norm(xx, *self.fitted_params)
		dydx = np.gradient(y, dx)
		dydxdx = np.gradient(dydx,dx)

		plt.plot(xx, dydx)
		# plt.plot(xx, dydxdx)

	def plot(self):
		self.plot_hist()
		self.plot_fit_sep()

		# Plot cut line
		plt.axvline(x=self.cut(), color="magenta", label="Cut")

		plt.xlabel("Loudness normalized [dB]")
		plt.ylabel("Number of counts")
		plt.title("Loudness distribution before cut")
		plt.legend()

		plt.savefig("Figures/loudness_dist_vanilla_" + str(self.i) + ".png")
		plt.show()

#################
# Read data
#################

filenames = glob('Data/*')
dataframes = [pd.read_csv(f, index_col='Time') for f in filenames]

N = 1

# data = pd.read_csv('Amplitudes_' + str(N) + '.csv',
#                    sep=',',
#                    # parse_dates=['Time'],
#                    index_col='Time')

# df = data.squeeze()

# # Filter data - rolling
# # df = filter(df)

# # df_cut = filter(df_cut)

# # Time cut on data - at the begining and at the end; to cut of
# # beeping
# if N == 1:
#     df_cut = df[20:235]
# elif N == 2:
#     df_cut = df[10:205]

# # Manual value cut on data - I supouse that popcorn sound was louder
# # than -44 dB

# # if N==1:
# # 	df_cut = df_cut[df_cut>-45.5]
# # if N==2:
# # 	df_cut = df_cut[df_cut>-48.5]

################
# Plot data + derivative
################

for df, i in zip(dataframes, range(1, 1+len(dataframes))):

	fig, axes = plt.subplots(2, 1, sharex=True)

	axes[0].set_ylabel("Amplitude  [dB]")
	axes[1].set_ylabel("Amplitude/second \
	[${}^{dB}/_\\mathrm{sec}$]")

	axes[0].set_title("Data")
	axes[1].set_title("Derivative")

	df.plot(label="Raw", legend=True, ax=axes[0])
	derivative(df).plot(label="Derivative", legend=True, ax=axes[1], style="g-")

	plt.savefig("Figures/data_" + str(i) + ".png")
	plt.show()

################
# Histogram before cuts
################

df_cut = []

for df, i in zip(dataframes, range(1, 1+len(dataframes))):

	x = biNorm(df, i)

	x.fit()

	x.norm()

	x.plot()

	df -= x.noise
	df_cut.append(df[df > x.cut()])

	i += 1

 # Sum of data
df_tot = pd.concat(dataframes)

# Plot histogram of all data

x = biNorm(df_tot, 0, [1,0,1,1,3,1])

x.fit()

# x.norm()

x.plot()

# df_tot -= x.noise


# # Automatic cut in the point where two Gaussians are equal
df_tot_cut = df_tot[df_tot > x.cut()]
df_tot_cut = df_tot_cut.dropna()

#################
# Histograms after cuts
#################

# hist = df_tot_cut.hist(bins=20, grid=False)

# plt.xlabel("Loudness [dB]")
# plt.ylabel("Number of counts")
# plt.title("Loudness distribution after cut")

# plt.savefig("Figures/loudness_dist_cut_" + str(N) + ".png")
# plt.show()

################
# Phase plot
################

# plt.scatter(df_tot.values,
#             derivative(df_tot).values,
#             marker='.',
#             color='green')

# plt.xlabel("Loudness [dB]")
# plt.ylabel("Loudness change [${}^{\\mathrm{dB}}/_\\mathrm{sec}$]")
# plt.title("Phase space diagram")

# plt.savefig("Figures/phase_space_" + str(N) + ".png")
# plt.show()

##################
# Time distribution of pops
##################

bins_ = 50

for df, i in zip([df_tot_cut, *df_cut], range(len(df_cut)+1)):

	df = df.dropna()

	# List of time of all pops
	pop_times = df.index.to_numpy()

	print("Number of pops %i" % len(pop_times))

	plt.hist(pop_times,
	         bins=bins_,
	         density=False)

	# Fitting gauss function
	(mu, sigma) = scipy.stats.norm.fit(pop_times)

	xmin, xmax = plt.xlim()
	x = np.linspace(xmin, xmax, 1000)
	norm = (xmax - xmin) / bins_ * len(pop_times)
	y = norm * scipy.stats.norm.pdf(x, mu, sigma)

	plt.plot(x, y, 'r--',
	         linewidth=2,
	         label="Fit results: \nmu = %.2f,  \nstd = %.2f" % (mu, sigma))

	plt.xlabel("Time [s]")
	plt.ylabel("Normalized counts")
	plt.title("Pops time distribution (tot = %i)" % len(pop_times))
	plt.legend()

	plt.savefig("Figures/time_dist_cut_" + str(i) + ".png")
	plt.show()

	i += 1