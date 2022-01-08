% Train CNN Netz

close all
clear 

% Einlesen der erkannten Schilder
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore

% 50% der Bilder zum Trainieren, 10% zum Validieren, 40% zum Testen
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, 0.5, 0.1, 0.4,'randomized');

% ----- Augmentation

outputSize = [227 227 3]; % [28 28 1];

% ----- Augmentation definieren und druchführen
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-30 30], ...
      'RandXTranslation', [-4 4], ....
'RandYTranslation', [-4 4])
trainingImageAugDS = augmentedImageDatastore(outputSize, trainingImageDS, 'DataAugmentation',imageAugmenter);
validationImageAugDS = augmentedImageDatastore(outputSize, validationImageDS, 'DataAugmentation',imageAugmenter);

% ----- einfaches DeepLearning Netzwerk definieren
layers = [
    imageInputLayer(outputSize) %imageInputLayer([28 28 1])

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
    'InitialLearnRate', 0.0005,...
    'MaxEpochs',300, ...                    
    'ValidationData', validationImageAugDS,...  % validationImageDS oder validationImageAugDS  ...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainingImageAugDS,layers,options);  % trainingImageDS oder trainingImageAugDS

% speichern des trainierten Netzes 
save netClassification.mat net;