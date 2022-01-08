function [trainingDataDS,validationDataDS, testDataDS, testDataTbl, signDatasetTbl] = LoadAndRandomizeData(inputSize)
% ----- Laden der Daten
imageDS = imageDatastore('Pictures_1024_768',"IncludeSubfolders",true,"LabelSource","foldernames");
dataVec = load('signDatasetGroundTruth.mat');  % 
signDatasetTbl = dataVec.signDataset;  % 1125*2 table
% ----- Aufteilung der Daten => 60% Training, 10% Validierung, 30% Testen  
rng(0)
shuffledIndicesVec = randperm(height(signDatasetTbl));
idxVec = floor(0.6 * height(signDatasetTbl));

trainingIdxVec = 1:idxVec;
trainingDataTbl = signDatasetTbl(shuffledIndicesVec(trainingIdxVec),:);

validationIdxVec = idxVec+1 : idxVec + 1 + floor(0.1 * length(shuffledIndicesVec) );
validationDataTbl = signDatasetTbl(shuffledIndicesVec(validationIdxVec),:);

testIdxVec = validationIdxVec(end)+1 : length(shuffledIndicesVec);
testDataTbl = signDatasetTbl(shuffledIndicesVec(testIdxVec),:);

% ----- Erzeugen der einzelnen Datastores, welche dann für das Training, die Validierung und zum Testen verwendet werden.
imdsTrainDS = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrainDS = boxLabelDatastore(trainingDataTbl(:,'sign'));

imdsValidationDS = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidationDS = boxLabelDatastore(validationDataTbl(:,'sign'));

imdsTestDS = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTestDS = boxLabelDatastore(testDataTbl(:,'sign'));

% ----- Kombination der Image- und LabelBox-Datastores
trainingDataDS = combine(imdsTrainDS,bldsTrainDS); % erzeugt 'CombinedDatastore
validationDataDS = combine(imdsValidationDS,bldsValidationDS);
testDataDS = combine(imdsTestDS,bldsTestDS);
data = read(trainingDataDS); %einlesen des DataStores

augmentedTrainingData = transform(trainingDataDS,@augmentData);
trainingDataDS = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationDataDS = transform(validationDataDS,@(data)preprocessData(data,inputSize));
end

