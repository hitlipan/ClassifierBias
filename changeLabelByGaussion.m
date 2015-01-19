function newC = changeLabelByGaussion(probs, mus, variances, tpr,  mapping_pred2Idx, lambda)
% probs : numClass * numCases
% mus : # of dimension is numClass
% variances : # of dimension is numClass
% tpr : percison = tp / (tp + fp)
% mapping_pred2Idx : mapping label to the position for liblinear
% lambda : the parameter trading off the primal probs and the external
%          probs.
[numClass, numCases] = size(probs);
newC = zeros(numCases, 1);
EPS = 1e-4;
for j = 1:numCases
      
     curProb = probs(:, j);
      t = curProb(mapping_pred2Idx(1:numClass));
      x = t';

      t = t - mus; t = t .^ 2; 
      t = t ./ (-2 * (variances + EPS)); t = exp(t); t = t ./ sqrt(variances + EPS);

       t = t .* tpr;
      t = t / (sum(t) + EPS);
       y = t';
       
      finalPro = lambda * t + (1 - lambda) * curProb(mapping_pred2Idx(1:numClass));
      [~, I] = max(finalPro);
      newC(j) = I;
end
end
