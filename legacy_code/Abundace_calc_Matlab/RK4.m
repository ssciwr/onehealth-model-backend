function v = RK4(func, func2, v, vars, step_t)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Runge-Kutta 4

    k1 = func(v, vars);
    disp(['k1 min: ', num2str(min(k1(:))), ', max: ', num2str(max(k1(:)))]);

    k2 = func(v + 0.5 .* k1 ./ step_t, vars);
    disp(['k2 min: ', num2str(min(k2(:))), ', max: ', num2str(max(k2(:)))]);

    k3 = func(v + 0.5 .* k2 ./ step_t, vars);
    disp(['k3 min: ', num2str(min(k3(:))), ', max: ', num2str(max(k3(:)))]);

    k4 = func(v + k3 ./ step_t, vars);
    disp(['k4 min: ', num2str(min(k4(:))), ', max: ', num2str(max(k4(:)))]);

    v1 = v + (k1 + 2*k2 + 2*k3 + k4) ./ (step_t * 6.0);
    disp(['v1 min: ', num2str(min(v1(:))), ', max: ', num2str(max(v1(:)))]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Check for negative numbers

    if any(v1 < 0) | any(v + 0.5 .* k1 ./ step_t < 0) | any(v + 0.5 .* k2 ./ step_t < 0) | any(v + k3 ./ step_t < 0)
        indices = union(union(union(find(v1 < 0), find(v + 0.5 .* k1 ./ step_t < 0)), find(v + 0.5 .* k2 ./ step_t < 0)), find(v + k3 ./ step_t < 0));
        v2 = log(v);
        FT2 = func2(v2, vars);
        v2 = v2 + FT2 ./ step_t;
        v1(indices) = exp(v2(indices));
    end

    v = v1;

end
