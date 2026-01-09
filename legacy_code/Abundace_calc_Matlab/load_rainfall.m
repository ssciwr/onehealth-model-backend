function PR = load_rainfall(pr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load rainfall data

PR = ncread(pr, 'tp');
PR = PR(:,:,:);     %Essentially, it's flattening the entire three-dimensional array into a single column vector. So, after this line of code, PR will be a column vector containing all the elements of the original three-dimensional array in a linear fashion. 
PR = double(PR);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
