close all
clear
doTraining = false;

% ----- load Data
imageDS = imageDatastore('Pictures_1024_768',"IncludeSubfolders",true,"LabelSource","foldernames");
dataVec = load('signDatasetGroundTruth.mat');  % 
signDataset = dataVec.signDataset;  % 1125*2 table

% ----- split the dataset into training, validation, and test sets.
% Select 60% of the data for training, 10% for validation, and the
% rest for testing the trained detector
rng(0)
shuffledIndices = randperm(height(signDataset));
idx = floor(0.6 * height(signDataset));

trainingIdx = 1:idx;
trainingDataTbl = signDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = signDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = signDataset(shuffledIndices(testIdx),:);

% ----- use imageDatastore and boxLabelDatastore to create datastores
% for loading the image and label data during training and evaluation.

imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'sign'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'sign'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'sign'));


% combine image and box label datastores.

trainingData = combine(imdsTrain,bldsTrain); % erzeugt 'CombinedDatastore
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);



% display one of the training images and box labels.

data = read(trainingData); %EVENTUELL ZUWEISEN
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,4);  % nur fuer Darstellung
figure
imshow(annotatedImage)



% ----- Create Faster R-CNN Detection Network
inputSize = [768 1024 3];

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
% Achtung: dieser DS wird nur zur Ermittlung der anchorBoxes verwendet

% % display one of the training images and box labels.
% while 1==0 %hasdata(preprocessedTrainingData)
%     data = read(preprocessedTrainingData);
%     I = data{1};
%     bbox = data{2};
%     annotatedImage = insertShape(I,'Rectangle',bbox);
%     annotatedImage = imresize(annotatedImage,4);  % nur fuer Darstellung
%     figure(1)
%     imshow(annotatedImage)
%     pause(0.100)
% end

% Auswahl der anchor boxes
%   Infos dazu: https://de.mathworks.com/help/vision/ug/estimate-anchor-boxes-from-training-data.html
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

% und das feature CNN
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
numClasses = width(signDataset)-1;    % also hier: 1, es sollen nur Autos erkannt werden

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% Netzwerk ansehen
% analyzeNetwork(lgraph) 


% Augmentierung
augmentedTrainingData = transform(trainingData,@augmentData);

trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);


if doTraining
    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
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

% Display the results.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% ----- Testing

testData = transform(testData,@(data)preprocessData(data,inputSize));

% Run the detector on all the test images.

detectionResults = detect(detector,testData,'MinibatchSize',1); 

% Evaluate the object detector using the average precision metric.

[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);
% The precision/recall (PR) curve highlights how precise a detector is at varying levels of recall. The ideal precision is 1 at all recall levels. The use of more data can help improve the average precision but might require more training time. Plot the PR curve.

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))


% ----- Helper functions

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


