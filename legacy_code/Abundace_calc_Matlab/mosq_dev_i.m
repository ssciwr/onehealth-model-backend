function T = mosq_dev_i(T)

%q = 1.695638e-04;
%T0 = 3.750303e+00;
%Tm = 3.553575e+01;

%T = q*T.*(T - T0 ).*((Tm - T).^(1/2));  % new function briere with coffiecint with initial data collection, for Sandra and Zia model

T = 50.1 - 3.574 * T + 0.069 * T.^2;

T = 1 ./ T;

end
