% selection D matrix in AHANM, size: n by k
% treat the data points as the same data point if they are close
function D = selectionM(X)
[~,~,rank] = uniquetol(X,.001);
k = length(uniquetol(X,.001));
n = length(X);
D = zeros(n,k);
for i = 1:n
    D(i,rank(i)) = 1;
end
D = sparse(D);