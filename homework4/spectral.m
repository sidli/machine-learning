function C = spectral(A, k)

D = diag(sum(A, 2));
L = D - A;
[Eig_vec, ~] = eig(L);
V = Eig_vec(:, 1:k);
idx = kmeans(V, k);

n = size(idx, 1);
C = cell(1, k);
for t = 1:k
    C{t} = (idx == t * ones(n, 1));
end

