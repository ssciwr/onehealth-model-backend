function [T, Tmean] = load_temp1(tmax, tmin, step_t)
Tmax = ncread(tmax, 't2m');
Tmin = ncread(tmin, 't2m');
Tmax = Tmax(:,:,:);  % This command is necessary to give the dimesnion and mateix form to extracted variable, Tmax is a 3D variable, Essentially, it's flattening the entire three-dimensional array into a single column vector. So, after this line of code, Tmax will be a column vector containing all the elements of the original three-dimensional array in a linear fashion. 
Tmin = Tmin(:,:,:);

%%
[x, y, z] = size(Tmax);

Tmean = 0.5 .* (Tmax + Tmin);

T = zeros(x, y, z * step_t);

%% Updating the mean temperature values to all 100 uniform temperature steps in a day, not governed by diurnal cycle of temperature

for t = 1:z*step_t
    td = ceil(t/step_t);        
    T(:,:,t) = Tmean(:,:,td);
end