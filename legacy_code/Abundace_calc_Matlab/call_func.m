function v_out = call_func(v, Temp, Tmean, LAT, CC, egg_activate, step_t)

diapause_lay = mosq_dia_lay(Tmean, LAT, step_t);
diapause_hatch = mosq_dia_hatch(Tmean, LAT, step_t);

ed_survival = mosq_surv_ed(Temp, step_t);
% Temp = Temp(:,:,90*step_t+1:end); % uncomment if previous year's temperature data was available for winter mortality calculation

v_out = zeros(size(v, 1), size(v, 2), 5, size(Temp, 3)/step_t);

for t = 1:size(Temp, 3)
    T = Temp(:,:,t);
    birth = mosq_birth(T);
    dev_j = mosq_dev_j(T);
    dev_i = mosq_dev_i(T);
    %dev_e = mosq_dev_e(T);
    dev_e = 1./7.1;      %original function of the model
    dia_lay = diapause_lay(:,:,ceil(t/step_t));
    dia_hatch = diapause_hatch(:,:,ceil(t/step_t));
    ed_surv = ed_survival(:,:,t);
    water_hatch = egg_activate(:,:,ceil(t/step_t));
    mort_e = mosq_mort_e(T);
    mort_j = mosq_mort_j(T);

    T = Tmean(:,:,ceil(t/step_t));
    mort_a = mosq_mort_a(T);

    vars = {ceil(t/step_t), step_t, Temp, CC, birth, dia_lay, dia_hatch, mort_e, mort_j, mort_a, ed_surv, dev_j, dev_i, dev_e, water_hatch};
    v = RK4(@eqsys, @eqsys_log, v, vars, step_t); % Runge-Kutta 4 method
    % v = FE(@eqsys, @eqsys_log, v, vars, step_t); % Forward Euler method

    if mod(t/step_t,365) == 200
        v(:,:,2) = 0;
    end

    if mod(t,step_t) == 0
        if mod(ceil(t/step_t),30) == 0
            disp(['MOY: ', num2str(t/step_t/30)]);
        end
        for j = 1:5
            %v_out(:,:,j,t/step_t) = v(:,:,j);
            v_out(:,:,j,t/step_t) = max(v(:,:,j), 0);  % if any abundance is negative it will make it zero
        end
    end
end

end
