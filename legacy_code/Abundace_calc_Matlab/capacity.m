function pr = capacity(pr, dens)

ALPHA = 1e-3;
BETA =  1e-5;
GAMMA = 9e-1;

LAMBDA =  1e6 * 625.0 * 100.0;

% Initialize a new variable to store the updated pr values
fprintf('PR size: %d \n',size(pr));
fprintf('dens size: %d \n',size(dens));

pr = double(pr);
dens = double(dens);

pr_new = pr;

% Calculate K(t) = l * (1-g)/(1-g^t) * A(t) with A(t) = g*A(t-1) + a*PR(t) + b*DENS(t)
pr_new(:,:,1) = ALPHA .* pr_new(:,:,1) + BETA .* dens;

for k = 2:size(pr_new,3)
    pr_new(:,:,k) = GAMMA .* pr_new(:,:,k-1) + ALPHA .* pr_new(:,:,k) + BETA .* dens;
end

for k = 2:size(pr_new,3)
    pr_new(:,:,k) = (1 - GAMMA) / (1 - GAMMA^k) * pr_new(:,:,k);
end

pr = pr_new .* LAMBDA;

end
