% Parameters
L = 10;        % Length of the domain
T = 1;         % Final time
Nx = 100;      % Number of spatial grid points
Nt = 100;      % Number of time steps
alpha = 0.01;  % Diffusion coefficient

% Spatial and temporal discretization
x = linspace(0, L, Nx);
t = linspace(0, T, Nt);
dx = x(2) - x(1);
dt = t(2) - t(1);

% Initial condition
u0 = sin(pi * x / L);

% Preallocate solution matrix
u = zeros(Nx, Nt);

% Set initial condition
u(:, 1) = u0;

% Adomian decomposition method
for n = 1:Nt-1
    % Compute Adomian polynomials
    A0 = u(:, n);
    A1 = -alpha * diff(u(:, n), 2) / dx^2;
    
    % Calculate coefficients
    a0 = A0;
    a1 = A1;
    
    % Compute solution term by term
    u(:, n+1) = u(:, n) + dt * (a0 + a1);
end

% Plot the results
figure;
surf(t, x, u');
title('Simulation of Burger''s Equation using Adomian Decomposition');
xlabel('Time');
ylabel('Space');
zlabel('u(x, t)');
