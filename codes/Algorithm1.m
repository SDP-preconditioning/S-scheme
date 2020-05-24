function [X,iter] = Algorithm1(alpha,rho,MSE,N,K,max_iter,C,X_cvx)

Lam = zeros(N+K,N+K);
Z = zeros(N+K,N+K);
S = ones(N+K);
S(1:N,1:N) = 1/alpha;
S(N+1:end,N+1:end) = alpha;
C11 = C(1:N,1:N);

% ADMM iterations
for iter = 1:max_iter           
    X22 = 1/(1+rho*alpha^2)*(C(N+1:end,N+1:end)+Lam(N+1:end,N+1:end)*alpha+rho*Z(N+1:end,N+1:end)*alpha);          
    X22(1:K+1:end) = ones(1,K); 
    X0 = 1/(1+rho)*(C(N+1:end,1:N) + Lam(N+1:end,1:N) +rho*Z(N+1:end,1:N) );   
    X = [C11, X0'; X0, X22];
            
    Q = S.*X-Lam/rho;    
    [V,D] = eig((Q+Q')/2);
    d = diag(D);
    d = max(d,0);      
    Z = V*diag(d)*V';

    Lam = Lam+rho*(Z-S.*X);

    if norm(X-X_cvx,'fro')^2/(N+K)^2 < MSE
        break;
    end 
end
end