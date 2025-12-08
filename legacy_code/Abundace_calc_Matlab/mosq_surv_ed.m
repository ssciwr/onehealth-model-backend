function T = mosq_surv_ed(T, step_t)

T = double(T);

ed_surv_bl = 1;

for k = 2:size(T, 3)
    T(:,:,k) = min(T(:,:,k-1:k),[],3);
end

% Uncomment the following line if previous year's data is available for winter mortality calculation
% T(:,:,1:90*step_t) = [];

T = ed_surv_bl .* 0.93 .* exp(-0.5 .* ((T - 11.68) ./ 15.67) .^ 6);

end

