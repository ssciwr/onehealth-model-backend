function T = mosq_mort_j(T)
  
T = double(T);

T = 0.977 * exp(-0.5 * ((T - 21.8) ./ 16.6) .^ 6);
T = -log(T);

end
