% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%                    TRAINING-REGION DETECTION
%------------------------------------------------------------------------%
% Mithilfe dieses Skriptes wird das ResNet50 belernt.
% Hierbei nutzen wir den uns zur Verfügung gestellten Datensatz, der
% Straßenverkehrsschilder. Im Anschluss kann ein besipielhaft belernter
% Datenpunkt ausgegeben werden.
%------------------------------------------------------------------------%

close all
clear

% ----- Zeige beispielhaftes Bild mit BoundingBox nach Training ----- %
showExample = false;

% ----- Definition inputSize & Rückgabe der Datastores ----- %
inputSizeVec = [448 448 3]; %[768 1024 3];
[trainingDataDS,validationDataDS, testDataDS, testDataTbl, signDatasetTbl] = LoadAndRandomizeData(inputSizeVec);

% ----- Augmentierung der Daten ----- %
augmentedTrainingData = transform(trainingDataDS,@augmentData); %u.a. horizontales spiegeln (siehe Funkton unten). Das ist vielleicht keine gute Idee !!
trainingDataDS = transform(augmentedTrainingData,@(data)preprocessData(data,inputSizeVec));
validationDataDS = transform(validationDataDS,@(data)preprocessData(data,inputSizeVec));

% ----- Erstellung des Faster Regionbased Convolutional Neuronal Network ----- %
preprocessedTrainingData = transform(trainingDataDS, @(data)preprocessData(data,inputSizeVec)); % Bilder werden auf die input Size skalliert

% ----- Ermittlung der Anchor-boxes ----- %
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors);

% ----- Feature CNN definieren ----- %
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
numClasses = width(signDatasetTbl)-1;    % Kategorien, die erkannt werden sollen ( hier 1: Schilder )
lgraph = fasterRCNNLayers(inputSizeVec,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% ----- Netzwerk ansehen ----- %
% analyzeNetwork(lgraph)

% ----- Trainingsoptionen ----- %
options = trainingOptions('sgdm',...
    'MaxEpochs',13,...
    'MiniBatchSize',1,...
    'InitialLearnRate',5e-4,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationDataDS,...
    'Plots','training-progress');

% ----- Trainieren des Netzes und anschließendes sichern ----- %
[detector, info] = trainFasterRCNNObjectDetector(trainingDataDS,lgraph,options);
save 'Neuronale_Netze/netDetectorResNet50.mat' detector;

% ----- Ausgabe eines beispielhaften Datenpunktes inkl. der BoundingBox ----- %
if showExample
    
    % ----- Ermittlung eines zufälligen Bildes aus dem Datensatz ----- %
    showIndx = floor(rand()*length(testDataTbl.imageFilename)) % Für zufälliges Bild
    Img = imread(testDataTbl.imageFilename{showIndx});   %I = imread(testDataTbl.imageFilename{3});
    [bboxes,scores] = detect(detector,Img);
    
    % ----- Ausgabe der Ergebnisse ----- %
    Img = insertObjectAnnotation(Img,'rectangle',bboxes,scores);
    figure
    imshow(Img)
end

