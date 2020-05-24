%This is a simple demo for the implementation  
%and effectiveness validation for Algorithm 1.
%% Data Generation
clear;clc;
% Basic parameters 
N = 200;             % Inut data dimension, take 
J = 3;               % Source number
rho = 1;             % ADMM penalty parameter
MSE = 1e-6;          % Mean squared error threshold
max_iter = 1e10;     % Maximum iteration number allowed

% Minimum Separation Condition for {\tau_j} with 1/N
for s_iter = 1:100  
    f = rand(J,1);   
    df = abs(bsxfun(@minus,f,f'));
    d1 = min(min(df+eye(J)));
    d2 = min(min(ones(J)-df));
    d = min(d1,d2);
    if d > 1/N
        break;
    end 
end    
A0 = exp(-1i*2*pi*f*(0:N-1));           % Fourier bases corresponds to J sources
c = randn(J,1);                         % Spikes amplitudes

Omega_size = 4*N/5;                     % Subset size
% Subsampling
index = randperm(Omega_size);
index = index(1:Omega_size);
index = sort(index);       

g_t = A0'*c;                            % Ground-truth from the superposition of J Fourier bases
y = g_t(index);                         % Partial observation

%% S-scheme embedded ADMM
alpha_s = 2*sqrt(N);    % S_scheme scaling factor
[x_s,iter_s] = Algorithm2(alpha_s,rho,MSE,N,max_iter,index,y,g_t);

alpha_1 = 1;            % No preconditioning
[x_1,iter_1] = Algorithm2(alpha_1,rho,MSE,N,max_iter,index,y,g_t);

% Iteration number comparison
iter_s
iter_1










