close all
clear



% ----- Ausgabe eines Bildes mit LabelBox ----- %
% inputSize definieren, mit der die Bilder eingelesen werden
% notwendig um Speicher der Grafikkarte nicht zu überladen
inputSize = [448 448 3]; %[768 1024 3];
[trainingDataDS,validationDataDS, testDataDS, testDataTbl, signDatasetTbl] = LoadAndRandomizeData(inputSize);
% ----- Augmentierung der Daten ----- %
augmentedTrainingData = transform(trainingDataDS,@augmentData); %u.a. horizontales spiegeln (siehe Funkton unten). Das ist vielleicht keine gute Idee !!
trainingDataDS = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationDataDS = transform(validationDataDS,@(data)preprocessData(data,inputSize));

% ----- Erstellung des Faster Regionbased Convolutional Neuronal Network ----- %
preprocessedTrainingData = transform(trainingDataDS, @(data)preprocessData(data,inputSize));
% Bilder werden auf die input Size skalliert

% ----- Ermittlung der Anchor-boxes ----- %
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors);

% ----- Feature CNN definieren ----- %
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
numClasses = width(signDatasetTbl)-1;    % Kategorien, die erkannt werden sollen ( hier 1: Schilder )

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

layersTransfer = featureExtractionNetwork.Layers(1:end-3);
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
        'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


% analyzeNetwork(lgraph) % ----- Netzwerk ansehen ----- %
   
options = trainingOptions('sgdm',...
    'MaxEpochs',13,...
    'MiniBatchSize',1,...
    'InitialLearnRate',5e-4,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationDataDS,...
    'Plots','training-progress');

% Train the Faster R-CNN detector.
[detector, info] = trainFasterRCNNObjectDetector(trainingDataDS,lgraph,options); %, ...
    %'NegativeOverlapRange',[0 0.3], ...
    %'PositiveOverlapRange',[0.6 1]);
% Trainiertes Netz abspeichern, um es in anderen Skripten verwenden zu
% können
save netDetectorResNet50.mat detector;

% ----- quick check/test ----- %
if showExample
    showIndx = floor(rand()*length(testDataTbl.imageFilename)) % Für zufälliges Bild
    Img = imread(testDataTbl.imageFilename{showIndx});   %I = imread(testDataTbl.imageFilename{3});
    [bboxes,scores] = detect(detector,Img);
    % ----- Ausgabe der Ergebnisse ----- %
    Img = insertObjectAnnotation(Img,'rectangle',bboxes,scores);
    figure
    imshow(Img)
end

