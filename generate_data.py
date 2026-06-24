# %Generate time series
# Generate multivariate time series, each of them a linear combination of
# sinusoids with different period length |f0| and variance |f0Var|.
# Furthermore, red noise realized as an AR(1) process with a noise level
# |NoiseLevel| is added. The AR(1) parametes are randomly chosen in
# |ARCoeff|.

# %clear all;

t=[1:10029];
t=[1:1029]';
t=[1:129)';
f0 = 1./[7.5  5.0 2.8  2.3];