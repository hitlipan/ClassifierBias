function  retVal = computeGau(mu, Sigma, x)

EPS = 1e-4;
[n] = length(x);
invSigma = inv(Sigma + EPS * eye(n, n));
Delta = (x - mu)' * invSigma * (x - mu);
Delta = -1/2 * Delta;
Delta = exp(Delta);
coef = 1 / sqrt(det(Sigma) + EPS);
retVal = coef * Delta;
end
