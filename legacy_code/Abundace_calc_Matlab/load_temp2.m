function [T, Tmean] = load_temp2(tmean, step_t)
Tmean = ncread(tmean, 't2m');
Tmean = double(Tmean(:,:,:));  % This command is necessary to give the dimesnion and mateix form to extracted variable, Tmax is a 3D variable, Essentially, it's flattening the entire three-dimensional array into a single column vector. So, after this line of code, Tmax will be a column vector containing all the elements of the original three-dimensional array in a linear fashion. 


%%
[x, y, z] = size(Tmean);

%Tmean = 0.5 .* (Tmax + Tmin);

T = zeros(x, y, z * step_t);
T = double(T);

%% Updating the mean temperature values to all 100 uniform temperature steps in a day, not governed by diurnal cycle of temperature

for t = 1:z*step_t
    td = ceil(t/step_t);        
    T(:,:,t) = Tmean(:,:,td);
end