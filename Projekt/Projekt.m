close all
clear

doTraining = false;

% ----- Laden der Daten
imageDS = imageDatastore('Pictures_1024_768',"IncludeSubfolders",true,"LabelSource","foldernames");
dataVec = load('signDatasetGroundTruth.mat');  % 
signDatasetTbl = dataVec.signDataset;  % 1125*2 table

% ----- Aufteilung der Daten => 60% Training, 10% Validierung, 30% Testen  
rng(0)
shuffledIndicesVec = randperm(height(signDatasetTbl));
idxVec = floor(0.3 * height(signDatasetTbl));

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

% ----- Ausgabe eines Bildes mit LabelBox
data = read(trainingDataDS); %EVENTUELL ZUWEISEN
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,4);  % nur fuer Darstellung
figure
imshow(annotatedImage)

% ----- Erstellung des Faster Regionbased Convolutional Neuronal Network
inputSize = [448 448 3];
preprocessedTrainingData = transform(trainingDataDS, @(data)preprocessData(data,inputSize));

% ----- Ermittlung der Anchor-boxes
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

% ----- und das feature CNN
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
numClasses = width(signDatasetTbl)-1;    % also hier: 1, es sollen nur Autos erkannt werden
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% ----- Netzwerk ansehen
% analyzeNetwork(lgraph) 

% ----- Augmentierung der Daten
augmentedTrainingData = transform(trainingDataDS,@augmentData);
trainingDataDS = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationDataDS = transform(validationDataDS,@(data)preprocessData(data,inputSize));
options = trainingOptions('sgdm',...
    'Plots','training-progress', ...
    'MaxEpochs',14,...
    'MiniBatchSize',1,...
    'InitialLearnRate',5e-4,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationDataDS);

if doTraining
    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingDataDS,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
    save netDetectorResNet50.mat detector;
else
    % Load pretrained detector for the example.
    load netDetectorResNet50.mat detector;
end

% ----- quick check/test
I = imread(testDataTbl.imageFilename{3});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

% ----- Ausgabe der Ergebnisse
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% ----- Testen des Neuronalen Netzes
testDataDS = transform(testDataDS,@(data)preprocessData(data,inputSize));

% ----- Teste das Neuronale Netz mit den Testbildern
detectionResults = detect(detector,testDataDS,'MinibatchSize',1); 

% ----- Auswertung des Detektors mithilfe der durschnittlichen Präzision
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testDataDS);
% The precision/recall (PR) curve highlights how precise a detector is at varying levels of recall. The ideal precision is 1 at all recall levels. The use of more data can help improve the average precision but might require more training time. Plot the PR curve.

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))


% ----- Hilfsfunktionen

function data = augmentData(data)
% Randomly flip images and bounding boxes horizontally.
tform = randomAffine2d('XReflection',true);
sz = size(data{1});
rout = affineOutputView(sz,tform);
data{1} = imwarp(data{1},tform,'OutputView',rout);

% Sanitize box data, if needed.
% data{2} = helperSanitizeBoxes(data{2}, sz);

% Warp boxes.
data{2} = bboxwarp(data{2},tform,rout);
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.
sz = size(data{1},[1 2]);
scale = targetSize(1:2)./sz;
data{1} = imresize(data{1},targetSize(1:2));

% Sanitize box data, if needed.
% data{2} = helperSanitizeBoxes(data{2}, sz);

% Resize boxes.
data{2} = bboxresize(data{2},scale);
end


