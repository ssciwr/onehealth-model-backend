function LAT = load_latitude(tmax)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load rainfall data
LAT = ncread(tmax, 'latitude');
LAT = double(LAT(:,:));              % Essentially, it's flattening the entire three-dimensional array into a single column vector. So, after this line of code, LAT will be a column vector containing all the elements of the original three-dimensional array in a linear fashion. 
end