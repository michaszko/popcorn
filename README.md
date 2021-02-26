# popcorn

Small project which has one goal -- to answer the question if distribution of
pops when making popcorn in microwave is Gaussian? The answer is **yes**.

## How it is done?

Data was gathered using an app called *PhyPhox* and module for the sound
registration. The data has a form of two columns: left one is time [in s] and
right one is sound pressure level [in dB]. Raw data is first plotted. Then this
data is taken and histogramized with respect to loudness. A bi-normal (i.e. sum
of two Gaussians) is fitted to obtained distribution. One Gauss is showing
white noise distribution and the other one - the pops distribution. Using this
histogram the cut is chosen and applied. Processed data is then histogramized
with respect to time. This is the final distribution we look for and it is
indeed Gaussian.

![final
distribution](https://raw.githubusercontent.com/michaszko/popcorn/master/Figures/time_dist_cut_0.png)
