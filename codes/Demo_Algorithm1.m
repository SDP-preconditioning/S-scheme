%This is a simple demo for the implementation  
%and effectiveness validation for Algorithm 1.
%% Data Generation
clear;clc;
% Basic parameters 
N = 21;              % Top-left block dimension
K = 21;              % Bottom-right block dimension
max_eig = 9;         % Maximum eigenvalue of the fixed block
rho = 25;            % ADMM penalty parameter
MSE = 1e-6;          % Mean Squared Error threshold
max_iter = 1e10;     % Maximum iteration number allowed
%rng(10);            % pseudo-random seed

% Uniformly generate the rest of the eigenvalues of the fixed block
rest_eig = rand(N-max_eig,1); % Non-negative eigenvalues  
rest_eig = rest_eig./norm(rest_eig,1)*(N-max_eig); % Normalization
x = [max_eig;rest_eig;zeros(max_eig-1,1)];
if max_eig == 1
    x = ones(N,1);
end

% Generate the observation data
C11 = gallery('randcorr',x); % Generate random correlation matrix with specified eigenvalues
C0 = 2*rand(K,N)-ones(K,N);
C22 = 2*rand(K,K)-ones(K,K);
C22 = (C22+C22')/2;
C22(1:K+1:end) = ones(1,K);
C = [C11,C0';C0,C22];

% Ground-truth generation for MSE comparison purpose
cvx_begin quiet
cvx_precision best
    variable X(N+K,N+K) 
    minimize ( 1/2*pow_pos(norm(X-C,'fro'),2) )
    subject to
        X == semidefinite(K+N,K+N); 
        X(1:N,1:N) == C11;
        diag(X(N+1:end,N+1:end)) == ones(K,1);
cvx_end
X_cvx = X;

%% S-scheme embedded ADMM
alpha_s = 0.05;   % S_scheme scaling factor
[X_s,iter_s] =  Algorithm1(alpha_s,rho,MSE,N,K,max_iter,C,X_cvx);

alpha_1 = 1;      % No preconditioning
[X_1,iter_1] =  Algorithm1(alpha_1,rho,MSE,N,K,max_iter,C,X_cvx);

% Iteration number comparison
iter_s
iter_1





