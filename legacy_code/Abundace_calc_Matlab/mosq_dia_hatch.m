function T = mosq_dia_hatch(T, LATU, step_t)

T = double(T);

%%%%%%%%%%%%%%

period = 7;
CPP = 11.25;
CTT = 11.0;
ratio_dia_hatch = 0.1;

%%%%%%%%%%%%%% Calculate mean temperature of the last 'period' days and compare to CTT_S

[x, y, z] = size(T);

for k = z:-1:period    % this loops starts from 365 get decreased by one in each iteration till period = 7, i.e. 365, 364 ... 7
    T(:,:,k) = mean(T(:,:,k-period+1:k), 3);  % Mean of last 7 days temperature stored in T itself by replacing original value of temperature at that day
end

T(T<CTT) = 0;

%%%%%%%%%%%%%% Calculate day length and compare to CPP_S

Phi = asin(0.39795*cos(0.2163108 + 2*atan(0.9671396*tan(0.0086*(mod((1:z),367) - 186)))));  %careful while applying for leap year in netcdf file having data more than one year

#LATU = double([-80, -10, 40, 80]);

for k = 1:y
    %lat = 61.5 - 12.5*k*1/y;     % change 61.5 according to max of latitude
    lat = LATU(k);  % according to changed formula
    
    %% Always use the real part of daylight otherwise the complex trigonmetric calculation might lead to complex number and the daylight will be zero
    daylight = real(24 - 24/pi * acos(sin(lat * pi / 180) * sin(Phi) ./ (cos(lat * pi / 180) * cos(Phi))));  % so daylight will be a array for 365 days
    %daylight(~isreal(daylight)) = 0; %%%%%%%%%%%%%% for regions above the arctic circle

    % The repmat create a matix with entries as daylight (365 entries) repeating 1 time in row, 1 time in column and 50 (long) times in 3D ( the dimesion of matrix will now be 1 * 365*50 and then the  permute do rearranging the dimensions so that the first dimension (originally the third dimension) becomes the third, the second becomes the first, and the third becomes the second 
    daylight = permute(repmat(daylight,1,1,x), [3 1 2]);    
    % The above permutation is necessary so that original T can be substituted with the daylight value
    % An easy to do this is also given in mosq_dia_lay 
    T_help = T(:,k,:);
    T_help(daylight < CPP) = 0;
    T(:,k,:) = T_help;
end

%%%%%%%%%%%%%%

T(isnan(T)) = 0;
T(T>0) = ratio_dia_hatch;

end
