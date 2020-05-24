function [x,iter] = Algorithm2(alpha,rho,MSE,N,max_iter,index,y,g_t)

  Z = zeros(N+1);
Lam = zeros(N+1);   
normalizer = diag(1./[N,2*N-2:-2:2]'); 
e1 = [1;zeros(N-1,1)];

% ADMM iterations
for iter = 1:max_iter
    x = Z(1:N,N+1)+Lam(1:N,N+1)/rho;
    x(index) = y;
    q = alpha*(normalizer*(T_adj( Z(1:N,1:N)+Lam(1:N,1:N)/rho )-alpha/(2*rho)*e1));
    t = (Z(N+1,N+1)-1/(2*rho*alpha)+Lam(N+1,N+1)/rho)/alpha;
    
    W = [toeplitz(q)/alpha,x;x',t*alpha];
    Q = W-Lam/rho;
    [V,D] = eig((Q+Q')/2);
    d = diag(D);
    d = max(d,0);
    Z = V*diag(d)*V';
    
    Lam = Lam+rho*(Z-W);
    
    if norm(x-g_t)^2/N < MSE
        break;
    end
    
end

function T_a = T_adj(Q)

    N = length(Q);
    T_a = zeros(N,1);    
    T_a(1) = trace(Q);
    for j = 1:N-1
        T_a(j+1) = 2*sum(diag(Q,j));
    end

end

end