function T = mosq_birth(T)

T = double(T);

T(T < 38.8) = 33.2 * exp(-0.5 * ((T(T < 38.8) - 70.3) ./ 14.1) .^ 2) .* (38.8 - T(T < 38.8)) .^ 1.5;
T(T >= 38.8) = 0;

end
