function [T, Tmean] = load_temp(tmax, tmean, tmin, step_t)
Tmax = ncread(tmax, 't2m');
Tmin = ncread(tmin, 't2m');
Tmax = Tmax(:,:,:);  % This command is necessary to give the dimesnion and matrix form to extracted variable, Tmax is a 3D variable, Essentially, it's flattening the entire three-dimensional array into a single column vector. So, after this line of code, Tmax will be a column vector containing all the elements of the original three-dimensional array in a linear fashion. 
Tmin = Tmin(:,:,:);
lati = ncread(tmax, 'latitude'); % To extract latitude from the file in order to calculate daylight wherever needed
lati = lati(:,:);   % This command is necessary to give the dimesnion and mateix form to extracted variable, since latitude is a 2D variable so a 2D matrix
%% 

Tmean = ncread(tmean, 't2m');
Tmean = Tmean(:,:,:);

Tmax_prev = 'previous_year_tx.nc';

if exist(Tmax_prev, 'file') == 2
    Tmax_prev = ncread(Tmax_prev, 'tx');
    Tmax_prev = Tmax_prev(:,:,end-90+1:end);
    Tmin_prev = 'previous_year_tn.nc';
    Tmin_prev = ncread(Tmin_prev, 'tn');
    Tmin_prev = Tmin_prev(:,:,end-90+1:end);

    Tmax = cat(3, Tmax_prev, Tmax);
    Tmin = cat(3, Tmin_prev, Tmin);
end

[x, y, z] = size(Tmax);

Phi = asin(0.39795 .* cos(0.2163108 + 2*atan(0.9671396 .* tan(0.0086*(mod(1:z,367) - 186)))));

sunrise = zeros(y, z);

for k = 1:y
    lat = lati(k);  % extracting the latitude
    sunrise(k,:) = real(12/pi .* acos(sin(lat * pi / 180) * sin(Phi) ./ (cos(lat * pi / 180) * cos(Phi)))); % updating the sunrise matrix accordingly
end

% This is the old formula for daylight calculation as mentioned in  Metalmann model, suitable only for England
%for lati = 1:y
 %   lat = 61.5 - 12.5*lati*1/y;    % change 61.5 according to max of latitude
    %lat = 75.375 - (lati-1)*0.25 - 0.375; 
    %lat = 75.5 - lati*50*1/y;
    %sunrise(y - lati + 1,:) = real(12/pi .* acos(sin(lat * pi / 180) * sin(Phi) ./ (cos(lat * pi / 180) * cos(Phi))));
%end


%% Diurnal temperature cycle incorporation

T = zeros(x, y, z * step_t);

for t = 1:z*step_t
    td = ceil(t/step_t);           % going from step size to the day
    hour = 24/step_t * (mod(t-1,step_t) + 1);    % hours of day
    for lat = 1:y
        sunny = sunrise(lat,td);
        Tmax_1 = Tmax(:,lat,td);
        Tmin_1 = Tmin(:,lat,td);
        if hour < sunny
            if t <= step_t        % if i-1 day data is not availble for Tmax calculation of last day
                T(:,lat,t) = (Tmax(:,lat,1) + Tmin(:,lat,1)) ./ 2 + (Tmax(:,lat,1) - Tmin(:,lat,1)) ./ 2 .* cos(pi * (hour+10) / (10 + sunny));
            else
                Tmax_0 = Tmax(:,lat,td-1);
                T(:,lat,t) = (Tmax_0 + Tmin_1) ./ 2 + (Tmax_0 - Tmin_1) ./ 2 .* cos(pi * (hour+10) / (10 + sunny));
            end
        elseif (hour < 14)
            T(:,lat,t) = (Tmax_1 + Tmin_1) ./ 2 - (Tmax_1 - Tmin_1) ./ 2 .* cos(pi * (hour - sunny) / (14 - sunny));
        else
            if t >= (z-1)*step_t
                T(:,lat,t) = (Tmax(:,lat,z) + Tmin(:,lat,z)) ./ 2 + (Tmax(:,lat,z) - Tmin(:,lat,z)) ./ 2 .* cos(pi * (hour-14) / (10 + sunny));
            else
                Tmin_2 = Tmin(:,lat,td+1);
                T(:,lat,t) = (Tmax_1 + Tmin_2) ./ 2 + (Tmax_1 - Tmin_2) ./ 2 .* cos(pi * (hour-14) / (10 + sunny));
            end
        end
    end
end

%end
