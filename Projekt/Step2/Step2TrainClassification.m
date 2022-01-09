% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%                    TRAINING-CLASSIFICATION
%------------------------------------------------------------------------%
% Mithilfe dieses Skriptes wird das AlexNet belernt.
% Hierbei nutzen wir den uns zur Verfügung gestellten Datensatz, der
% Straßenverkehrsschilder. Im Anschluss wird die erreichte Präzision
% des Netzes ausgegeben.
%------------------------------------------------------------------------%

close all
clear

% ----- Variablen zur Verteilung der Daten ----- %
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten

% ----- Variablen für die Trainingsparameter des Netzes ----- %
initialLearnRate = 0.0005;                          %Defaultwert 0.01
maxEpochs = 300;                                    %Defaultwert 30
miniBatchSize = 128;                                %Defaultwert 128
validationFrequency = 30;                           %Defaultwert 50

% ----- Einlesen der erkannten Schilder in einen Datastore ----- %
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');
fprintf("Anzahl Bilder: %d\n", length(imageDS.Labels));

rng(7);
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');
fprintf("Trainingsmenge Anzahl Elemente: %d  Test(Validierungs)menge: %d\n", length(trainingImageDS.Labels), length(validationImageDS.Labels));

% ----- Deklaration des AlexNet ----- %
net = alexnet;
inputSizeVec = [227 227 3];

% ----- Augmentation definieren und durchführen ----- %
imageAugmenter = imageDataAugmenter( ...
                'RandRotation', [-45 45], ...
                'RandXTranslation', [-4 4], ....
                'RandYTranslation', [-4 4]);
trainingImageAugDS = augmentedImageDatastore(inputSizeVec, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(inputSizeVec, validationImageDS, 'DataAugmentation',imageAugmenter);

% ----- Andwendung des Transfer Learnings ----- %
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImageDS.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
        'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% ----- Trainingsoptionen ----- %
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',initialLearnRate, ... 
    'ValidationData',validationImageAugDS, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience', 5, ...
    'Verbose',false, ...
    'Plots','training-progress');

% ----- Training des Netzes und anschließende Sicherung ----- %
netTransfer = trainNetwork(trainingImageAugDS,layers,options);
save 'Neuronale_Netze\netAlexClassification.mat' netTransfer;

% ----- Berechnung und Ausgabe der erreichten Präzision ----- %
YPred = classify(netTransfer, validationImageAugDS);
accuracy = mean(YPred == validationImageDS.Labels)
