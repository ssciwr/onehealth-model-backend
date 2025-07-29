## List of terms

- LCD: Local Climate Data
- DTR: Diurnal Temperature Range



## How the code works
### Name: `aedes_albopictus_model.m`

1. Create variables to assign the prefix of each dataset that will be used
   ```matlab
   tmean_prefix = 'ERA5land_global_t2m_daily_0.5_';
   pr_prefix = 'ERA5land_global_tp_daily_0.5_';
   dens_prefix = 'pop_dens_';
   ```

2. Create a for loop to iterate over years. The following operations are done in each loop:
    a. The name for each dataset is completed with the year and extension:
    ```matlab
    tmean = [tmean_prefix, num2str(years), '.nc'];
    pr = [pr_prefix, num2str(years), '.nc'];
    dens = [dens_prefix, num2str(years), '_global_0.5.nc'];
    ```
    
    b. Log the year that is being processed:
    ```matlab
    year = years; % change years according to file name
    disp(['Year in Process: ', num2str(years)]);
    ```

    c. Run the model that calculates the differential equations. This model receives the following datasets
    - `tmean`: mean temperature dataset
    - `pr`: precipitation dataset
    - `dens`: population density dataset
    - `year`: year of the dataset
    
    ```matlab
    %model_run(tmax,tmin,tmean, pr,dens,year);
    model_run(tmean, pr,dens,year);            % for tmean only with NO DTR (Tmax or Tmin is not present)
    disp(['Year done: ', num2str(years)]);
    ```  
3. End the program

### Name: `model_run.m`
1. Create a variable name for the output file.
```matlab
outfile = strcat('Mosquito_abundance_Global_', num2str(year), '.nc')      % change the output file name as required

if exist(outfile) == 2
    delete(outfile)
end
```
2. Copy the dataset that contains the precipitation (one of the ERA5*.nc) and store this copy using the variable name for the output file.
```matlab
copyfile(pr, outfile);   % overwriting the file gives us advantage that we can loose the ;latitude and longitude information of Tmax,
% Tmin file and only work with time varible in 3rd dimension and then rewrite this over same nc file
```
3. Rename the internal variable (`pr`) of this copy with the name `adults`. It means, we have a new .nc file that contains
`longitude`, `latitude` and the variable name `adults`. With this we can just overwrite the variable adult with the calculations we intend to do.
```matlab
ncid = netcdf.open(outfile,'NC_WRITE');
netcdf.reDef(ncid)
netcdf.renameVar(ncid,3,'adults');
netcdf.endDef(ncid);
netcdf.close(ncid);
```

4. Defines a delta time step.
TODO: get more information about this time step
```matlab
step_t = 10;
```

5. Load variables from each dataset involved (`tmean`, `pr`, `dens`)
   a. Load Temperature variables and applies some preprocessing steps with `load_temp2(tmean, step_t)` function
   ```matlab
   [Temp, Tmean] = load_temp2(tmean, step_t);       % Without DTR if Tmean is only available, no Tmax or Tmin
   ```
   b. Load Population density variables and applies some preprocessing steps with `load_hdp(dens)`
   ```matlab
   DENS = load_hpd(dens);
   ```
   c. Load Precipitation variable with `load_rainfall(pr)` function
   ```matlab
   PR = load_rainfall(pr);
   ```
   d. Load Latitude variable from the `tmean` dataset with the function `load_latitude(tmean)`
   ```matlab
   LAT = load_latitude(tmean);
   ```

6. Calculate **Juvenile Carrying Capacity**: \(K_{L}(W,P)\)
\(W\): Rainfall accumulation, this the information we get from `PR`
\(P\): Human density, this is the information we get from `DENS`
```matlab
CC = capacity(PR, DENS);
```
- file associated: `capacity.m`
- Equation in Supplement Information: 14

7. Calculate **Hatching fraction depending in human density and rainfall**: \(Q(W,P)\)
$$
Q(W,P) = (1 - E_{rat}) \left( \frac{(1+E_{0})e^{\left(-E_{var}(W(t)-E_{opt})\right)^2}}{e^{\left( -E_{var} (W(t)-E_{opt})^2 \right)}} \right) + E_{rat} \left( \frac{E_{dens}}{E_{dens} + e^{-E_{fac}P}} \right)
$$

Where:
\(W\): Precipitation, we get this information from `PR`
\(P\): Human density, we get this information from `DENS`
\(E_{opt}\) = 8;
\(E_{var}\) = 0.05;
\(E_{0}\) = 1.5;
\(E_{rat}\)= 0.2;
\(E_{dens}\) = 0.01;
\(E_{fac}\) = 0.01;
```matlab
egg_active = water_hatch(PR, DENS);
```
- file associated: `water_hatch.m`
- Equation in Supplement Information: 13

8. Create a **Vector Initial population**: \(V_{0}\)
```matlab
previous = 'no_previous';
v0 = load_initial(previous, size(Temp));
```
- File associated: `load_initial.m`
- TODO: Ask about the name and meaning of this variable.

9. Calculate the parameters for each time step and run ODEs
```matlab
v = call_func(v0, Temp, Tmean, LAT, CC, egg_active, step_t);
```
- Input variables:
  - initial vector: `v0`
  - temperature: `Temp`
  - mean temperature: `Tmean`
  - latitude: `LAT`
  - juvenile carrying capacity: `CC`
  - hatching fraction depending in human density and rainfall: `egg_active`
  - time step: `step_time`
- File associated: `call_func.m`

10.  Once all the variables are calculated, write to the output file
```matlab
ncwrite(outfile,'adults', permute(v(:,:,5,:),[1,2,4,3]));
```

### Name: `call_func.m`

Description: This file the main file that contains the ODEs to calculate the 6 differential equations
proposed in the paper.

- Input Variables:
  - `v`
  - `Temp` 
  - `Tmean`
  - `LAT`
  - `CC`
  - `egg_activate`
  - `step_t`
  
- Process:
  
1. Calculate the **Diapause Lay**

- File associated: `mosquito_dia_lay.m`
- TODO: ask about the equation.

2. Calculate the **Diapause Hatch**
- File associated: `msoquito_dia_hatch.m`
- TODO: ask about the equation

3. Calculate **Diapausing egg mortality rate** : \(m_{Ed}(T)\)
$$
m_{Ed}(T) = -\ln\left(  0.955 \left(-0.5 \left(\frac{T-18.8}{21.53}\right)^6 \right) \right)
$$

- Input Variables: 
  - `Temp`
  - `step_t`
- File associated: `mosq_surv_ed.m`
- Comments: hardcoded values do not correspond to the values in the paper.

4. Create a n-dimensional matrix `v_out` full of zeros.

5. Create a for loop that iterates over the time:
  5.1. Creates a temperattures vector
    ```octave
    T = Temp(:,:,t);
    ```
  5.2. Calculates the Mosquito Birth
    - Inputs:
      - `T`
    - File associated:
      - `mosq_birth.m`
    - Comments:
      - TODO: This equation is not present in the suplement material.
    - 
    ```octave
    T = Temp(:,:,t);
    ```
    5.3. Calculates the Mosquito development j
    - Inputs:
      - `T`
    - File associated:
      - `mosq_dev_j.m`
    - Comments:
      - TODO: This equation is not present in the suplement material.
    - 
    ```octave
    dev_j = mosq_dev_j(T);
    ```
    5.3. Calculates the Mosquito development i
    - Inputs:
      - `T`
    - File associated:
      - `mosq_dev_i.m`
    - Comments:
      - TODO: This equation is not present in the suplement material.
    - 
    ```octave
    dev_j = mosq_dev_i(T);
    ```
    5.4. Other calculations I do not understand
    ```octave
    dev_e = 1./7.1;      %original function of the model 
    dia_lay = diapause_lay(:,:,ceil(t/step_t));
    dia_hatch = diapause_hatch(:,:,ceil(t/step_t));
    ed_surv = ed_survival(:,:,t);
    water_hatch = egg_activate(:,:,ceil(t/step_t));
    mort_e = mosq_mort_e(T);
    mort_j = mosq_mort_j(T);
    
    T = Tmean(:,:,ceil(t/step_t));
    mort_a = mosq_mort_a(T);
    ```
    5.5. ODE
    ```octave
    vars = {ceil(t/step_t), step_t, Temp, CC, birth, dia_lay, dia_hatch, mort_e, mort_j, mort_a, ed_surv, dev_j, dev_i, dev_e, water_hatch};
    v = RK4(@eqsys, @eqsys_log, v, vars, step_t); % Runge-Kutta 4 method
    % v = FE(@eqsys, @eqsys_log, v, vars, step_t); % Forward Euler method
    ```
    5.6. Additional calculations
    ```octave
    if mod(t/step_t,365) == 200
        v(:,:,2) = 0;
    end
    
    if mod(t,step_t) == 0
        if mod(ceil(t/step_t),30) == 0
            disp(['MOY: ', num2str(t/step_t/30)]);
        end
        for j = 1:5
            %v_out(:,:,j,t/step_t) = v(:,:,j);
            v_out(:,:,j,t/step_t) = max(v(:,:,j), 0);  % if any abundance is negative it will make it zero
        end
    end

    ```