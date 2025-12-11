pkg load netcdf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load packages
%% FOR LCD_2025, we don't use any Tmin or Tmax data so we ignore the DTR calculation
%% Lines related to DTR input and calculation are commented out

%('/Users/pratik/Desktop/ae_albopictus_model')

% change tmax, tmin, pr and dens according to your data
% go to load_temp and load_rainfall function to change the variable according to your nc file
%see to line 39 to 92 to seee futher instructions
% go to mosqiuto diap, hatch and load temp and change the latitude extent according to your max lat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Don't forget to put ; after each line otherwise code will print the output of that line every time you run the code if its a loop then all the output.

%%%%%!!!!!!!! Check the dimension of each climate and population matrix file it should be same for each input variable !!!!!!!!!!!!%%%%%%%%%%%%

% for a single run of one partiular year

%tmax = 'tmax1.nc';
%tmin = 'tmin1.nc';
%pr = 'tp1.nc';
%dens = 'pop_dens_2015_EU_0.25.nc';

%year = 2015


%% for creating a loop of many years one by one

% Common file names
%tmax_prefix = 'era5land_tmax_EU_daily_0.25_';
%tmin_prefix = 'era5land_tmin_EU_daily_0.25_';
tmean_prefix = 'ERA5land_global_t2m_daily_0.5_';
pr_prefix = 'ERA5land_global_tp_daily_0.5_';
dens_prefix = 'pop_dens_';

% Loop over years
for years = 2024:2024                                     % set the years according to data
    % Construct file names for the current year
    %tmax = [tmax_prefix, num2str(years), '.nc'];      % every entry should be in "" for string format
    %tmin = [tmin_prefix, num2str(years), '.nc'];
    tmean = [tmean_prefix, num2str(years), '.nc'];
    pr = [pr_prefix, num2str(years), '.nc'];
    dens = [dens_prefix, num2str(years), '_global_0.5.nc'];

year = years;                                                   % change years according to file name

disp(['Year in Process: ', num2str(years)]);

%model_run(tmax,tmin,tmean, pr,dens,year);
model_run(tmean, pr,dens,year);         % for tmean only with NO DTR (Tmax or Tmin is not present)

disp(['Year done: ', num2str(years)]);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



