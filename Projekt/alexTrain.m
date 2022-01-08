% Trainieren des AlexNet
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
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames')
fprintf("Anzahl Bilder: %d\n", length(imageDS.Labels));

rng(7);
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');
fprintf("Trainingsmenge Anzahl Elemente: %d  Test(Validierungs)menge: %d\n", length(trainingImageDS.Labels), length(validationImageDS.Labels));

% Das Alexnet laden und in net speichern
net = alexnet;

% Augmentation definieren und durchführen
outputSize = [227 227 3];

imageAugmenter = imageDataAugmenter( ...
                'RandRotation', [-45 45], ...
                'RandXTranslation', [-4 4], ....
                'RandYTranslation', [-4 4]);
trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);


% Definierunbg des Netzwerks
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImageDS.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
        'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Trainingsoptionen
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',initialLearnRate, ... 
    'ValidationData',validationImageAugDS, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(trainingImageAugDS,layers,options);

% besser waere ein abschlissendes Testen mit´neuen´ Daten
%  und nicht mit den Validierungsdaten 
YPred = classify(netTransfer, validationImageAugDS);
accuracy = mean(YPred == validationImageDS.Labels)

% Speichern des trainierten Netzes
save netAlexClassification.mat netTransfer;