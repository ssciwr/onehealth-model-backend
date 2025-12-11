%function v = model_run(tmax, tmin, tmean, pr, dens, year)
function v = model_run(tmean, pr, dens, year)    % running for tmean only without DTR

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create output file

	outfile = strcat('Mosquito_abundance_Global_', num2str(year), '.nc')      % change the output file name as required

	if exist(outfile) == 2
		delete(outfile)
    end

	copyfile(pr, outfile);   % overwriting the file gives us advantage that we can loose the ;latitude and longitude information of Tmax, Tmin file and only work with time varible in 3rd dimension and then rewrite this over same nc file

	ncid = netcdf.open(outfile,'NC_WRITE');
	netcdf.reDef(ncid)
	netcdf.renameVar(ncid,3,'adults');
	netcdf.endDef(ncid);
	netcdf.close(ncid);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	step_t = 10;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%[Temp, Tmean] = load_temp(tmax, tmean, tmin, step_t);
    %[Temp, Tmean] = load_temp1(tmax, tmin, step_t);       % Without DTR
    [Temp, Tmean] = load_temp2(tmean, step_t);       % Without DTR if Tmean is only available, no Tmax or Tmin
    DENS = load_hpd(dens);
	PR = load_rainfall(pr);
    LAT = load_latitude(tmean);



    CC = capacity(PR, DENS);
    egg_active = water_hatch(PR, DENS);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create v0

	previous = 'no_previous';

	v0 = load_initial(previous, size(Temp));

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calc parameters for each time step and run ODEs

	v = call_func(v0, Temp, Tmean, LAT, CC, egg_active, step_t);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Write to outfile

	ncwrite(outfile,'adults', permute(v(:,:,5,:),[1,2,4,3]));


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot suitability for example

   % figure(1);
    % Calculate the ratio
    %ratio = v(:, :, 2, end) ./ v(:, :, 2, 1);

    % Avoid taking the logarithm of zero or negative values
   % ratio(ratio <= 0) = NaN;
    % Display the image
   % imagesc(rot90(log(ratio)), [-5, 5]);
    % Display the image
  	%imagesc(rot90(log(v(:, :, 2, end) ./ v(:, :, 2, 1))), [-10, 10]); % Set caxis limits here
	%colorbar();
	%colormap(jet);
	%output_filename = strcat('example_', num2str(year), '.png');
    %saveas(1, output_filename);



end
