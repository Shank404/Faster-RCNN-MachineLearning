% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
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
% Mithilfe dieses Skriptes wird das CNN belernt.
% Hierbei nutzen wir den uns zur Verfügung gestellten Datensatz, der
% Straßenverkehrsschilder.
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
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');

% ----- Angabe der Input-Size ----- %
inputSizeVec = [227 227 3];

% ----- Augmentation definieren und durchführen ----- %
imageAugmenter = imageDataAugmenter( ...
                'RandRotation', [-30 30], ...
                'RandXTranslation', [-4 4], ....
                'RandYTranslation', [-4 4]);
trainingImageAugDS = augmentedImageDatastore(inputSizeVec, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(inputSizeVec, validationImageDS, 'DataAugmentation',imageAugmenter);

% ----- Definition der Schichten im Neuronalen Netz ----- %
layers = [
    imageInputLayer(inputSizeVec)

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

% ----- Trainingsoptionen ----- %
options = trainingOptions('sgdm',...
    'InitialLearnRate', initialLearnRate,...
    'MaxEpochs',maxEpochs, ...                    
    'MiniBatchSize',miniBatchSize,...
    'ValidationData', validationImageAugDS,...  % validationImageDS oder validationImageAugDS
    'ValidationFrequency',validationFrequency,...
    'Verbose',false,...
    'Plots','training-progress');

% ----- Training des Netzes und anschließende Sicherung ----- %
net = trainNetwork(trainingImageAugDS,layers,options);  % trainingImageDS oder trainingImageAugDS
save 'Neuronale_Netze/netClassification.mat' net;