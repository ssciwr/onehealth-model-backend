function FT = eqsys_log(v, vars)

    [t, step_t, Temp, CC, birth, dia_lay, dia_hatch, mort_e, mort_j, mort_a, ed_surv, dev_j, dev_i, dev_e, water_hatch] = vars{};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Differential equations immatures

    FT(:,:,1) = v(:,:,5) .* birth .* (1 .- dia_lay) ./ v(:,:,1) .- (mort_e .+ water_hatch .* dev_e);
    FT(:,:,2) = v(:,:,5) .* birth .* dia_lay ./ v(:,:,2) .- water_hatch .* dia_hatch;
    FT(:,:,3) = water_hatch .* dev_e .* v(:,:,1) ./ v(:,:,3) .+ water_hatch .* dia_hatch .* ed_surv .* v(:,:,2) ./ v(:,:,3) .- (mort_j .+ dev_j) .- v(:,:,3) ./ CC(:,:,t);
    FT(:,:,4) = 0.5 * dev_j .* v(:,:,3) ./ v(:,:,4) .- (mort_a .+ dev_i);
    FT(:,:,5) = dev_i .* v(:,:,4) ./ v(:,:,5) .- mort_a;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Replace NaN values

    FT(isnan(-FT)) = - v(isnan(-FT)) * step_t;

endfunction
