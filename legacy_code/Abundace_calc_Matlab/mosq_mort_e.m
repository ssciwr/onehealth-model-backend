function T = mosq_mort_e(T)

T = 0.955 * exp(-0.5 * ((T - 18.8) ./ 21.53) .^ 6);
T = -log(T);

end
