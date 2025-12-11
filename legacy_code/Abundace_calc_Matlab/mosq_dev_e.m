function T = mosq_dev_e(T)

q = 0.0001246068;
T0 = -7.0024634748;
Tm = 34.1519214674;

T = q*T.*(T - T0 ).*((Tm - T).^(1/2));  % new function briere with coffiecint with initial data collection, for Sandra and Zia model

%T = 50.1 - 3.574 * T + 0.069 * T.^2;

%T = 1 ./ T;

end