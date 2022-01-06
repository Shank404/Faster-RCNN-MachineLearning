% Kategorisierung der erkannten Schilder

close all
clear


% falls die imageDataStore Directories nicht vorhanden sind: einmalig: 
% generateImageDataStoreFilesFunc('Train', 1);  % 1 => 166; 50 => 5000
% generateImageDataStoreFilesFunc('Full', 50);  

% Einlesen der erkannten Schilder
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore

% 50% der Bilder zum Trainieren, 10% zum Validieren, 40% zum Testen
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, 0.5, 0.1, 0.4,'randomized');

% ----- Augmentation

outputSize = [227 277 3]; % [28 28 1];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-50,50], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5])
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
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% ----- Training

options = trainingOptions('sgdm',...
    'MaxEpochs',300, ...                    
    'ValidationData', validationImageDS,...  % validationImageDS oder validationImageAugDS  ...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainingImageDS,layers,options);  % trainingImageDS oder trainingImageAugDS

predictedLabels = classify(net, validationImageDS);
accuracy = mean(predictedLabels == validationImageDS.Labels)

% ----- Anwendung des trainierten Netzwerkes: Ein Bild erkennen
% testImageDS = imageDatastore('Full','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore

% ein Bild entnehmen
[T, info] = read(testImageDS);
str = cellstr(info.Label)
image(T)
classify(net, T)

% ----- Anwendung des trainierten Netzwerkes: Gegen den Gesamtdatensatz
%                                             testen
predictedLabels = classify(net, testImageDS);
accuracy = mean(predictedLabels == testImageDS.Labels)


function generateImageDataStoreFilesFunc(name, absAngle)

[XTrain, YTrain, angles] = digitTrain4DArrayData;
% XTrain is 4-D Array: H-by-W-by-C-by-N array, where H is the
%     height and W is the width of the images, C is the number
%     of channels, and N is the number of images
% YTrain is Categorical vector containing the labels for each observation
% angles is Numeric vector containing the angle of rotation in
%   degrees for each image.
%
% XTrain konkret: 28 (height) * 28 (width) * 1 (channel) * 5000 (images)
%
name

for n = 0:9
    a= strcat(name, "\Ziffer", string(n))
    mkdir(char(a))    
end

n = 0;
for k=1:5000
    if (abs(angles(k)) <= absAngle)
        n = n + 1;
        strZif = char('Ziffer' + string(YTrain(k)));
        strArg = strcat(name, '\', strZif, '\', strZif, string(k), '.png')
        a = XTrain(:,:,1,k);
        imwrite(a, char(strArg));
    end
end

fprintf('Picture count: %d\n', n);
end

% layersTommy = [  imageInputLayer([28 28 1])
%     
%             convolution2dLayer(5,20)        % 5*5 Kernel, 20 Kernels
%             batchNormalizationLayer
%             reluLayer()
%             
%             maxPooling2dLayer(2,'Stride',2)
%             
%             fullyConnectedLayer(10)
%             softmaxLayer()
%             classificationLayer()
%          ];
