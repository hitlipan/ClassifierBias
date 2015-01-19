function main
% driver of ClassifierBias
% Written by Li Pan <lipan@hit.edu.cn>
% All the features stored under 'features', this program 
%   read the featuers and classify


imageName = '15Scenes';
feaDir = fullfile('features', imageName);
% tranverse the feature folder, get every feature path and their 
%     corresponding labels

feaDatabase = collectFeaInfo(feaDir); 
trNum = 100;

trIdx = [];
tsIdx = [];
% store the position of  training data and test data
for jj = 1:feaDatabase.numClass
    idxLabel = find(feaDatabase.labels == jj);
    num = length(idxLabel);
    idxRand = randperm(num);
    trIdx = [trIdx; idxLabel(idxRand(1:trNum))];
    tsIdx = [tsIdx; idxLabel(idxRand(trNum + 1:end))];
end
fprintf('Training num per class: %d\n', trNum);
load(feaDatabase.fpaths{1});
dFea = length(fea.att);
trFea = zeros(dFea, length(trIdx));
trLabel = zeros(length(trIdx), 1);
tsFea = zeros(dFea, length(tsIdx));
tsLabel = zeros(length(tsIdx), 1);
for ii = 1:length(trIdx)
    load(feaDatabase.fpaths{trIdx(ii)});
    trFea(:, ii) = fea.att;
    trLabel(ii) = fea.label;
end

trFea = zeros(dFea, length(trIdx));
trLabel = zeros(length(trIdx), 1);
tsFea = zeros(dFea, length(tsIdx));
tsLabel = zeros(length(tsIdx), 1);

for ii = 1:length(trLabel)

  load(feaDatabase.fpaths{trIdx(ii)});  % Load training features
  trFea(:, ii) = fea.att;      % Get the training data
  trLabel(ii) = fea.label;     % Get the training label
end

% Train the model by softmax
options.maxIter = 30;
inputSize = length(trFea(:, 1));
lambda = 1e-4;
softmaxModel = softmaxTrain(inputSize, feaDatabase.numClass, lambda, trFea, trLabel, options);
[pred] = softmaxPredict(softmaxModel, trFea);
% compute the precision for every class using training data
acces = zeros(feaDatabase.numClass, 1);
for jj = 1:feaDatabase.numClass
    idxes1 = find(pred(:) == jj);
    idxes2 = find(trLabel(:) == jj);
    acces(jj) = length(intersect(idxes1, idxes2))  / length(idxes1);
end

% predict
for jj = 1:length(tsLabel)
    load(feaDatabase.fpaths{tsIdx(jj)});
    tsFea(:, jj) = fea.att;
    tsLabel(jj) = fea.label;
end

[pred] = softmaxPredict(softmaxModel, tsFea);
acc = mean(tsLabel(:) == pred(:));
fprintf(1, 'Accuracy with out tpr: %0.3f%%\n', acc * 100);

tpr = acces;
[pred] = softmaxPredict(softmaxModel, tsFea, tpr);
acc = mean(tsLabel(:) == pred(:));
fprintf('Accuracy with tpr : %0.3f%%\n', acc * 100);
end
