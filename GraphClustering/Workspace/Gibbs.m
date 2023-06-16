a = 1;
b = 1;
A = 1;
T = 100; % T clusterings approximately sampled from the posterior.
X = [1 1 0 0; 1 1 0 1; 0 0 0 1; 0 1 1 1];

% function Z = irm(X,T,a,b,A)
N = size(X,1); z = true(N,1); Z = cell(T,1); % Initialization
for t = 1:T % For each Gibbs sweep
    for n = 1:N % For each node in the graph
        nn = [1:n-1 n+1:N]; % All indices except n
        K = size(z,2); % No. of components
        m = sum(z(nn,:))'; M = repmat(m,1,K); % No. of nodes in each component
        M1 = z(nn,:)'*X(nn,nn)*z(nn,:)- ... % No. of links between components
        diag(sum(X(nn,nn)*z(nn,:).*z(nn,:))/2);
        M0 = m*m'-diag(m.*(m+1)/2) - M1; % No. of non-links between components
        r = z(nn,:)'*X(nn,n); R = repmat(r,1,K); % No. of links from node n
        logP = sum([betaln(M1+R+a,M0+M-R+b)-betaln(M1+a,M0+b) ... % Log probability of n belonging
        betaln(r+a,m-r+b)-betaln(a,b)],1)' + log([m; A]); % to existing or new component
        P = exp(logP-max(logP)); % Convert from log probability
        i = find(rand<cumsum(P)/sum(P),1); % Random component according to P
        z(n,:) = false; z(n,i) = true; % Update assignment
        z(:,sum(z)==0) = []; % Remove any empty components
    end
    Z{t} = z; % Save result
end