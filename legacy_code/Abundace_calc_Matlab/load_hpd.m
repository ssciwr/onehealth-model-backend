function DENS = load_hpd(dens)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load human population density data

DENS = ncread(dens, 'dens');
DENS = double(DENS(:,:,:));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
