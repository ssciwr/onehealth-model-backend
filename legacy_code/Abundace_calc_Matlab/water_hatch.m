function pr = water_hatch(pr, dens)

e_opt = 8.0;
e_var = 0.05;
e_0 = 1.5;
e_rat = 0.2;
e_dens = 0.01;
e_fac = 0.01;

% TODO: remove 
dens = double(dens);
pr = double(pr);

% Calculate egg hatching rate impacted by human activity
dens = e_dens ./ (e_dens + exp(-e_fac .* dens));

% Calculate egg hatching rate according to model by Abdelrazec and Gumel (2017)
pr = (1 + e_0) * exp(-e_var .* (pr - e_opt) .^ 2) ./ (exp(-e_var .* (pr - e_opt) .^ 2) + e_0);

pr = (1 - e_rat) .* pr + e_rat .* repmat(dens, 1, 1, size(pr,3));



end
