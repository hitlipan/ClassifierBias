function main
% driver of ClassifierBias
% Written by Li Pan <lipan@hit.edu.cn>
% All the features stored under 'features', this program 
%   read the featuers and classify


imageName = '15Scenes';
feaDir = fullfile('features/bovw', imageName);
% tranverse the feature folder, get every feature path and their 
%     corresponding labels

feaDatabase = collectFeaInfo(feaDir); 
trNum = 30;
viNum = 30;

nRounds = 5;
Acces1 = zeros(nRounds, 1);
Acces2 = zeros(nRounds, 1);

MULTI_GAUSSION =  true;

for nite = 1:nRounds
trIdx = [];
viIdx = [];
tsIdx = [];

% store the position of  training data and test data
for jj = 1:feaDatabase.numClass
    idxLabel = find(feaDatabase.labels == jj);
    num = length(idxLabel);
    idxRand = randperm(num);
    trIdx = [trIdx; idxLabel(idxRand(1:trNum))];
    viIdx = [viIdx; idxLabel(idxRand(trNum + 1:trNum + viNum))];
    tsIdx = [tsIdx; idxLabel(idxRand(trNum + viNum + 1:end))];
end
fprintf('Training num per class: %d\n', trNum);
load(feaDatabase.fpaths{1});
dFea = length(fea.att);
trFea = zeros(dFea, length(trIdx));
trLabel = zeros(length(trIdx), 1);
viFea = zeros(dFea, length(viIdx));
viLabel = zeros(length(viIdx), 1);
tsFea = zeros(dFea, length(tsIdx));
tsLabel = zeros(length(tsIdx), 1);
for ii = 1:length(trIdx)
    load(feaDatabase.fpaths{trIdx(ii)});
    trFea(:, ii) = fea.att;
    trLabel(ii) = fea.label;
end

for jj = 1:length(viIdx)
     load(feaDatabase.fpaths{viIdx(jj)});
     viFea(:, jj) = fea.att;
     viLabel(jj) = fea.label;
end

options = ['-s 0 -c 5 -q'];
modelFea = train(double(trLabel), sparse(trFea'), options);
[pred, ~, probs] = predict(trLabel, sparse(trFea'), modelFea, '-b 1');
probs = probs';
[~, I] = max(probs, [], 1);
% Compute mapping for liblinear, labels v.s. position
mapping_pred2Idx = zeros(feaDatabase.numClass, 1);
mapping_pred2Idx(pred) = I;
[pred, ~, probs] = predict(viLabel, sparse(viFea'), modelFea, '-b 1');

%%%%%%%%%%%%%%%%%%%%%
% Train a classifier to test whether a given pros is reliable.
%%%%%%%%%%%%%%%%%%%%
probLabels = zeros(length(pred), 1);
probLabels(find(pred(:) == viLabel(:))) = 1;
revisedProbs = probs(:, mapping_pred2Idx(1:feaDatabase.numClass));
M = train(double(probLabels), sparse(revisedProbs), options);

probs = probs';
if MULTI_GAUSSION
    EPS = 1e-4;
    mus1 = cell(feaDatabase.numClass, 1);
    Sigmas1 = cell(feaDatabase.numClass, 1);
    mus2 = mus1;
    Sigmas2 = Sigmas1;
    revisedProbs = probs(mapping_pred2Idx(1:feaDatabase.numClass), :);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %    mean and convariance of gaussian for right prediction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:feaDatabase.numClass
        idx1 = find(viLabel == j);
        idx2 = idx1(find(pred(idx1) == j));
        mus1{j} = sum(revisedProbs(:, idx2), 2);
        mus1{j} = mus1{j} / (length(idx2) + EPS);
        Sigmas1{j} = (revisedProbs(:, idx2) - repmat(mus1{j}, [1, length(idx2)])) * (revisedProbs(:, idx2) - repmat(mus1{j}, [1, length(idx2)]))';
        Sigmas1{j} = Sigmas1{j} / (length(idx2) + EPS);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   mean and covariance for wrong prediction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      for j = 1:feaDatabase.numClass
        idx1 = find(viLabel == j);
        
        idx2 = idx1(find(pred(idx1) ~= j));
        mus2{j} = sum(revisedProbs(:, idx2), 2);
        mus2{j} = mus2{j} / (length(idx2) + EPS);
        Sigmas2{j} = (revisedProbs(:, idx2) - repmat(mus2{j}, [1, length(idx2)])) * (revisedProbs(:, idx2) - repmat(mus2{j}, [1, length(idx2)]))';
        Sigmas2{j} = Sigmas2{j} / (length(idx2) + EPS);
      end
end


fid = fopen('result.txt', 'a+');
fprintf(fid, '\n\n %s :\n', imageName);
fprintf(fid, 'training num : %f\n', trNum);
acc = mean(viLabel(:) == pred(:));

% predict
for jj = 1:length(tsLabel)
    load(feaDatabase.fpaths{tsIdx(jj)});
    tsFea(:, jj) = fea.att;
    tsLabel(jj) = fea.label;
end

[c, ~, probs] = predict(tsLabel, sparse(tsFea'), modelFea, '-b 1');
[~, I] = max(probs', [], 1);

acc = mean(tsLabel(:) == c(:));    Acces1(nite) = acc;
%fprintf(1, 'Accuracy without tpr: %0.3f%%\n', acc * 100);
    
    probs = probs';
    revisedProbs = probs(mapping_pred2Idx(1:feaDatabase.numClass), :);

    %Here first we use the classifier to determine whether it correctly 
    %  classify the sample
    newLabels = zeros(length(tsLabel), 1);
    newLabels(find(tsLabel(:)==c(:))) = 1;
    [p, ~, pprobs] = predict(newLabels, sparse(revisedProbs'), M, '-b 1');
    part1 = [];
    for j = 1:length(tsLabel)
        curProb = revisedProbs(:, j);
        newProbs = [];
        for k = 1:feaDatabase.numClass
            newProbs = [newProbs, computeGau(mus1{k}, Sigmas1{k}, revisedProbs(:, j))];
        end
        newProbs = newProbs / (sum(newProbs) + EPS);
        part1 = [part1, newProbs'];
    end
    part2 = [];
    for j = 1:length(tsLabel)
     
        curProb = revisedProbs(:, j);
        newProbs = [];
        for k = 1:feaDatabase.numClass
            newProbs = [newProbs, computeGau(mus2{k}, Sigmas2{k}, revisedProbs(:, j))];
        end
        newProbs = newProbs / (sum(newProbs) + EPS);
        part2 = [part2, newProbs'];
    end

    rightPros = pprobs(:, 1);
    wrongPros = pprobs(:, 2);
    rightPros = rightPros';
    wrongPros = wrongPros';
    rightPros = repmat(rightPros, [feaDatabase.numClass, 1]);
    wrongPros = repmat(wrongPros, [feaDatabase.numClass, 1]);
    part1 = part1 .* rightPros;
    part2 = part2 .* wrongPros;
    for j = 1:length(tsLabel)
        part1(:, j) = part1(:, j) / sum(part1(:, j));
        part2(:, j) = part2(:, j) / sum(part2(:, j));
    end
    
    lambda = 0.9;
    finalProbs = lambda *  revisedProbs + (1 - lambda)*(1*part1 + 1 * part2);
    [~, newC] = max(finalProbs, [], 1);
    acc = mean(tsLabel(:) == newC(:));   Acces2(nite) = acc;
     

fclose(fid);

end
fprintf(1, 'Accuracy without tpr : %0.3f%%, std : %0.3f%%\n', mean(Acces1) * 100, std(Acces1) * 100);
fprintf(1, 'Accuracy with tpr : %0.3f%%, std : %0.3f%%\n', mean(Acces2) * 100, std(Acces2) * 100);

end
