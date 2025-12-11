function T = mosq_dev_j(T)

%q = 5.116230e-5;
%T0 = 7.628991e+00;
%Tm = 4.086981e+01;

%T = q*T.*(T - T0 ).*((Tm - T).^(1/2));  % new function briere with coffiecint with initial data collection, for Sandra and Zia model
T = 82.42 - 4.87 * T + 0.08 * T.^2;   %old parameter description in original model

T = 1 ./ T;
 
end
