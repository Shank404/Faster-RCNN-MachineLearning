% Train CNN Netz
close all
clear 

% Variablen zur Verteilung der Daten
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten

% Variablen für die Trainingsparameter des Netzes
initialLearnRate = 0.0005;                          %Defaultwert 0.01
maxEpochs = 300;                                    %Defaultwert 30
miniBatchSize = 128;                                %Defaultwert 128
validationFrequency = 30;                           %Defaultwert 50

% Einlesen der erkannten Schilder
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');

% Aufteilung der Bilder in Trainingdatastores, Validationdatastores und Testdatastores
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');

% ----- Augmentation
outputSize = [227 227 3];

% ----- Augmentation definieren und druchführen
imageAugmenter = imageDataAugmenter( ...
                'RandRotation', [-30 30], ...
                'RandXTranslation', [-4 4], ....
                'RandYTranslation', [-4 4]);
trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);

% ----- einfaches DeepLearning Netzwerk definieren
layers = [
    imageInputLayer(outputSize)

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer
];

% ----- Training
options = trainingOptions('sgdm',...
    'InitialLearnRate', initialLearnRate,...
    'MaxEpochs',maxEpochs, ...                    
    'MiniBatchSize',miniBatchSize,...
    'ValidationData', validationImageAugDS,...  % validationImageDS oder validationImageAugDS
    'ValidationFrequency',validationFrequency,...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainingImageAugDS,layers,options);  % trainingImageDS oder trainingImageAugDS

% speichern des trainierten Netzes 
save netClassification.mat net;