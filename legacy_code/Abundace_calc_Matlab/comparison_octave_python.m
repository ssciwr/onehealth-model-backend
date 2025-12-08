% DESCRIPTION:
% This script runs the `aedes_albopictus_model` using dummy data in Octave,
% with the primary goal of comparing each model function and output step-by-step
% against the Python implementation. It loads temperature, rainfall, and population
% density datasets, computes all intermediate and final model variables, and
% prints their dimensions and values for manual verification. This process ensures
% that the Octave and Python versions produce consistent and reproducible results.
%
% MAIN STEPS:
% 1. Loads datasets for temperature, rainfall, and population density.
% 2. Computes and #displays initial conditions and first-level model outputs
%    (carrying capacity, water hatching).
% 3. Computes and #displays second-level model outputs (diapause, survival, etc.).
% 4. Iterates over time steps, printing slices and shapes of all relevant
%    variables, and advances the model state using a Runge-Kutta 4 ODE solver.
% 5. Stores and #displays the output, ensuring no negative abundances.
%
% PURPOSE:
% - To verify that the Octave implementation produces results consistent with
%   the Python version, using the same inputs and comparing intermediate and
%   final outputs.
% - To aid debugging and ensure reproducibility by providing detailed output
%   at each step.
% ------------------------------------------------------------------------------
% Author: Edwin Carreño.
% Affiliation: Software Scientific Center - 2025

% -- Octave preparation
clear
clc

% -- Load netcdf4 library
pkg load netcdf

%------------------------------------------------------------------------------
%----    Entering to the entry point: `aedes_albopictus_model` function    ----
%------------------------------------------------------------------------------

%% Define paths
% -- Large datasets
% tmean = "ERA5land_global_t2m_daily_0.5_2024.nc";
% pr = "ERA5land_global_tp_daily_0.5_2024.nc";
% dens = "pop_dens_2024_global_0.5.nc";

% -- Dummy Datasets
tmean = "dataset_temperature_dummy_octave.nc";
pr = "dataset_rainfall_dummy_octave.nc";
dens = "dataset_population_dummy_octave.nc";

% Time step for the ODE
step_t = 10;

% 1. Load Datasets
% -- Temperatures
[Temp, Tmean] = load_temp2(tmean, step_t);

% -- Population
DENS = load_hpd(dens);

% -- Rainfall
PR = load_rainfall(pr);

% -- Latitude
LAT = load_latitude(tmean);

% -- Initial conditions
previous = 'no_previous';
v0 = load_initial(previous, size(Temp));

disp(['Dimension of Initial Conditions: ', mat2str(size(v0))]);
disp(['Dimension of Latitudes: ', mat2str(size(LAT))]);
disp(['Dimension of Population density: ', mat2str(size(DENS))]);
#disp(['Dimension of Temperature: ', mat2str(size(Temp))]);
#disp(['Dimension of Temperature mean: ', mat2str(size(Tmean))]);


%----------------------------------------------------------------------------
%-----------    Entering to the first level: model_run` function    ---------
%----------------------------------------------------------------------------
% Note: verify manually with results given in python

% Verify carrying capacity
CC = capacity(PR, DENS);
#disp(['Dimension of CC: ', mat2str(size(CC))]);
#disp(CC)

% Verify water hatching
egg_active = water_hatch(PR, DENS);
#disp(['Dimension of water hatching: ', mat2str(size(egg_active))]);
#disp(egg_active)

% Verify Initial conditions
#disp(['Dimension of initial conditions: ', mat2str(size(v0))]);
#disp(v0)

%----------------------------------------------------------------------------
%-----    Entering to the second level: `call_function` function     --------
%----------------------------------------------------------------------------
% Verify diapause lay
diapause_lay = mosq_dia_lay(Tmean, LAT, step_t);
#disp(['Dimension of diapause_lay: ', mat2str(size(diapause_lay))]);
#disp(diapause_lay)

% Verify diapause_hatch
diapause_hatch = mosq_dia_hatch(Tmean, LAT, step_t);
#disp(['Dimension of diapause_hatch: ', mat2str(size(diapause_hatch))]);
#disp(diapause_hatch)

% Verify ed survival
ed_survival = mosq_surv_ed(Temp, step_t);
#disp(['Dimension of ed survival: ', mat2str(size(ed_survival))]);
#disp(ed_survival)

# Assign
v = v0;

# Create output array
v_out = zeros(size(v, 1), size(v, 2), 5, size(Temp, 3)/step_t);

for t = 1:size(Temp, 3)
    #if t == 12  # Just to run a portion of the code
    #  break;
    #end

    disp(['--- Time Step ', mat2str(t), ' ---']);

    # Line a. Verify this slice
    T = Temp(:,:,t);
    #disp(['Dimension of temperature slice: ', mat2str(size(T))]);
    #disp(T)

    # Line b. Verify this slice
    birth = mosq_birth(T);
    #disp(['Dimension of birth slice: ', mat2str(size(birth))]);
    #disp(birth)

    # Line c. Verify this slice
    dev_j = mosq_dev_j(T);
    #disp(['Dimension of dev_j slice: ', mat2str(size(dev_j))]);
    #disp(dev_j)

    # Line d. Verify this slice
    dev_i = mosq_dev_i(T);
    #disp(['Dimension of dev_i slice: ', mat2str(size(dev_i))]);
    #disp(dev_i)

    # Line e. Verify this slice
    %%dev_e = mosq_dev_e(T);
    dev_e = 1./7.1;      %original function of the model
    #disp(['Dimension of dev_e slice: ', mat2str(size(dev_e))]);
    #disp(dev_e)

    # Line f. Verify this slice
    idx_time = ceil(t/step_t);
    #disp(['\tidx time: ', mat2str(idx_time)]);

    dia_lay = diapause_lay(:,:,idx_time);
    #disp(['Dimension of dia_lay slice: ', mat2str(size(dia_lay))]);
    #disp(dia_lay)

    # Line g. Verify this slice
    dia_hatch = diapause_hatch(:,:,idx_time);
    #disp(['Dimension of dia_hatch slice: ', mat2str(size(dia_hatch))]);
    #disp(dia_hatch)

    # Line h. Verify this slice
    ed_surv = ed_survival(:,:,t);
    #disp(['Dimension of ed_surv slice: ', mat2str(size(ed_surv))]);
    #disp(ed_surv)

    # Line i. Verify this slice
    water_hatch = egg_active(:,:,idx_time);
    #disp(['Dimension of water_hatch slice: ', mat2str(size(water_hatch))]);
    #disp(water_hatch)

    # Line j. Verify this slice
    mort_e = mosq_mort_e(T);
    #disp(['Dimension of mort e slice: ', mat2str(size(mort_e))]);
    #disp(mort_e)

    # Line k. Verify this slice
    mort_j = mosq_mort_j(T);
    #disp(['Dimension of mort j slice: ', mat2str(size(mort_j))]);
    #disp(mort_j)

    # Line l. Verify this slice
    T = Tmean(:,:,ceil(t/step_t));
    #disp(['Dimension of T line k slice: ', mat2str(size(T))]);
    #disp(T)

    # Line m. Verify this slice
    mort_a = mosq_mort_a(T);
    #disp(['Dimension of mort a slice: ', mat2str(size(mort_a))]);
    #disp(mort_a)

    # Line n. Verify the variables that will passed to the ODE solver step
    vars = {
      ceil(t/step_t),
      step_t,
      Temp,
      CC,
      birth,
      dia_lay,
      dia_hatch,
      mort_e,
      mort_j,
      mort_a,
      ed_surv,
      dev_j,
      dev_i,
      dev_e,
      water_hatch
    };

    for i = 1:length(vars)
        var = vars{i};
        if isstruct(var)
            #disp(['  Var ', num2str(i-1), ' is a struct']);
        else
            #disp(['  Var ', num2str(i-1), ' shape: ', mat2str(size(var))]);
        end
    end

    # Line o. Verify the ODE step
    %v = FE(@eqsys, @eqsys_log, v, vars, step_t); % Forward Euler method
    v = RK4(@eqsys, @eqsys_log, v, vars, step_t); % Runge-Kutta 4 method
    #disp(['Dimension of ODE step: ', mat2str(size(v))]);
    #disp(v)

     disp(mod(ceil(t/step_t),30))

    if mod(t/step_t,365) == 200

        v(:,:,2) = 0;
    end

    if mod(t,step_t) == 0
        if mod(ceil(t/step_t),30) == 0
          disp(['MOY: ', num2str(t/step_t/30)]);
        end

        for j = 1:5
            v_out(:,:,j,t/step_t) = max(v(:,:,j), 0);  % if any abundance is negative it will make it zero
        end
    end
end

disp(size(v_out))
for i = 1:4
  disp(['out adult time: ', mat2str(i)]);
  disp(v_out(:, :, 5, i));
end




