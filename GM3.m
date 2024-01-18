% k by k G matrix, same point if two are close, .001
function G = GM3(X)
A = uniquetol(X,.001);
k = length(A);
G = zeros(k);
G(1,1) = 1/A(1);
G(2,2) = 1/(A(2)-A(1));
G(2,1) = -1/A(1)-1/(A(2)-A(1));
for i = 3:k
    G(i,i-2) = 1/(A(i-1)-A(i-2));
    G(i,i-1) = -((1/(A(i-1)-A(i-2)))+(1/(A(i)-A(i-1))));
    G(i,i) = 1/(A(i)-A(i-1));
end
G = sparse(G);