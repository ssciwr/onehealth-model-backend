function T = mosq_dia_lay(T, LATU,  step_t)

ratio_dia_lay = 0.5;

% Calculate daylength and compare to CPP, for latitudes between 49°N and 61.5°N
[x, y, z] = size(T);


Phi = asin(0.39795*cos(0.2163108 + 2*atan(0.9671396*tan(0.0086*(mod(1:z,367) - 186)))));  %careful while applying for leap year in netcdf file having data more than one year

T = double(T);
LATU = double(LATU);

#LATU = double([-80, -10, 40, 80]);

for k = 1:y
    %lat = 61.5 - 12.5*k*1/y;     % change 61.5 according to max of latitude
    lat = LATU(k);  % according to changed formula
    %disp(lat)
    %disp(Phi)
    
    %% Always use the real part of daylight otherwise the complex trigonmetric calculation might lead to complex number and the daylight will be zero
    daylight = real(24 - 24/pi * acos(sin(lat * pi / 180) * sin(Phi) ./ (cos(lat * pi / 180) * cos(Phi))));
    %disp(lat)
    %disp(sin(lat * pi / 180.0))
    %disp(sin(lat * pi / 180.0) * sin(Phi))
    %disp(cos(lat * pi / 180.0) * cos(Phi))
    
    %disp(daylight)
    
    daylight(~isreal(daylight)) = 0; % for regions above the arctic circle, some of daylight values could be complex also for region above arctic circle
    CPP = 10.058 + 0.08965 * lat;
    daylight(daylight > CPP) = 0;
    T(:, k, :) = repmat(daylight,x,1); % Corrected assignment

end

% No diapause induction in the first half of the year
for k = 0:z/365-1
    T(:,:,1+k*365:183+k*365) = 0;
end

T(T>0) = 1;

T = T * ratio_dia_lay;


end
