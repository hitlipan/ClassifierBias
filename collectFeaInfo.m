function  feaDatabase = collectFeaInfo(feaDir)
% collect the information of features, feature path and coresponding labels
%  feaDatabase.fpaths  path for every feature
%  feaDatabase.labels  coresponding labels for every feature

classfolders = dir(feaDir);
feaDatabase = struct;
feaDatabase.fpaths = {};
feaDatabase.labels = [];
feaDatabase.numClass = 0;
feaDatabase.numFeas = 0;
feaDatabase.classNames = {};

numLabel = 0;
for i = 1:length(classfolders)
    curFolderName = classfolders(i).name;
    if ~strcmp(curFolderName, '.') & ~strcmp(curFolderName, '..')
        numLabel = numLabel + 1;
        
        feaDatabase.numClass = feaDatabase.numClass + 1; % update feaDatabase.numClass
        feaDatabase.classNames = [feaDatabase.classNames, curFolderName]; %update feaDatabase.classNames
        
        path = fullfile(feaDir, curFolderName);
        feaInstances = dir(path);
        
        % add every instance in class i to feaDatabase
        for j = 1:length(feaInstances)
            curInstanceName = feaInstances(j).name;
            if ~strcmp(curInstanceName, '.') & ~strcmp(curInstanceName, '..')
                feaDatabase.numFeas = feaDatabase.numFeas + 1;  % update feaDatabase.numFeas;
                fullpath = fullfile(path, curInstanceName);
                feaDatabase.fpaths = [feaDatabase.fpaths, fullpath]; %update feaDatabase.fpaths
                feaDatabase.labels = [feaDatabase.labels; numLabel]; % update feaDatabase.labels
            end
        end
    end
end

end
