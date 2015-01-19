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
VIS = 0;
trNum = 30;
viNum = 30;
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
% Train the model by softmax
inputSize = length(trFea(:, 1));

options = ['-s 0 -c 5 -q'];
modelFea = train(double(trLabel), sparse(trFea'), options);

[pred, ~, probs] = predict(viLabel, sparse(viFea'), modelFea, '-b 1');

probs = probs';
[~, I] = max(probs, [], 1);
mapping = zeros(feaDatabase.numClass, 1);
mapping(pred) = I;
%[pred] = predict(trLabel, sparse(trFea'), modelFea);

% Learn the dictionary B, 
labelMatrix = zeros(feaDatabase.numClass, viNum);
I = 0:15:(viNum - 1)* 15;
viLabel = mapping(viLabel);
labelMatrix(I + viLabel) =  1;
B = labelMatrix * probs' * inv(probs * probs');


% compute the precision for every class using validation data
acces = zeros(feaDatabase.numClass, 1);

if VIS
  maxProbs = cell(feaDatabase.numClass, 1);
  wrongMaxProbs = cell(feaDatabase.numClass, 1);
  supposedMaxProbs = cell(feaDatabase.numClass, 1);
  I = [];
  supposedI = [];
  supposedProbs = [];
end

for jj = 1:feaDatabase.numClass
    idxes1 = find(pred(:) == jj);
    idxes2 = find(viLabel(:) == jj);
    inIdx = intersect(idxes1, idxes2);
    acces(jj) = length(inIdx)  / length(idxes1);
  if VIS
    maxProbs{jj} = probs(mapping(jj), inIdx);
    wrongMaxProbs{jj} = probs(mapping(jj), setdiff(idxes1, idxes2));
    supposedMaxProbs{jj} = probs(mapping(jj), setdiff(idxes2, idxes1));
    % find all wrong samples  
    r = setdiff(idxes2, inIdx);
    I = [I; r]; 
    supposedI = [supposedI, ones(1, length(setdiff(idxes2, inIdx))) * jj];    
  end
end
if VIS
     supposedI = mapping(supposedI);
     supposedProbs = probs(:, I); 
     save('maxProbs.mat', 'maxProbs');
     save('wrongMaxProbs.mat', 'wrongMaxProbs');
     save('supposedMaxProbs.mat', 'supposedMaxProbs');
     save('supposedI.mat', 'supposedI');
     save('supposedProbs.mat', 'supposedProbs');
end

acces'

fid = fopen('result.txt', 'a+');
fprintf(fid, '\n\n %s :\n', imageName);
fprintf(fid, 'training num : %f\n', trNum);
acc = mean(viLabel(:) == pred(:));
fprintf(1, 'Accuracy in vilidataion data : %0.3f%%\n', acc * 100);
fprintf(fid, 'Accuracy in validation data %0.3f%%\n', acc * 100);



% predict
for jj = 1:length(tsLabel)
    load(feaDatabase.fpaths{tsIdx(jj)});
    tsFea(:, jj) = fea.att;
    tsLabel(jj) = fea.label;
end

[c, ~, probs] = predict(tsLabel, sparse(tsFea'), modelFea, '-b 1');
acc = mean(tsLabel(:) == c(:));
fprintf(1, 'Accuracy with out tpr: %0.3f%%\n', acc * 100);
fprintf(fid, 'Accuracy without tpr : %0.3f%%\n', acc * 100);
[~, I] = max(probs', [], 1);
% compute the mapping for liblinear
mapping = zeros(feaDatabase.numClass, 1);
mapping(I) = c;

tpr = acces;
fprintf(fid, '%f ', acces);



lambda = 0.5;
[I] = changeProWithTpr(probs, tpr, lambda, mapping);
acc = mean(tsLabel(:) == I(:));
fprintf(1, 'Accuracy with tpr : %0.3f%%\n', acc * 100);
fprintf(fid, 'lambda = %f\tAccuracy with tpr : %0.3f%%\n', lambda, acc * 100);
fclose(fid);
end
