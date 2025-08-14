
% Install NETCDF4 package
pkg install "https://downloads.sourceforge.net/project/octave/Octave%20Forge%20Packages/Individual%20Package%20Releases/netcdf-1.0.18.tar.gz"

% load netcdf package
pkg load netcdf

% Paths to files

tmean = "temperature_dummy.nc"
pr = "pr_dummy.nc"
dens = "dense_dummy.nc"

% Variables

step_t = 10

% Load files
[Temp, Tmean] = load_temp2(tmean, step_t);

DENS = load_hpd(dens);

PR = load_rainfall(pr);

LAT = load_latitude(tmean);

CC = capacity(PR, DENS);
egg_active = water_hatch(PR, DENS);

previous = 'no_previous';

v0 = load_initial(previous, size(Temp));

v = call_func(v0, Temp, Tmean, LAT, CC, egg_active, step_t);