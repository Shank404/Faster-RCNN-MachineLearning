%
% Matlab Example 
%   Get Started with Transfer Learning
%
% todo:
%
% 12.10.2018
%
close all
clear

% Variablen zur Verteilung der Daten
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten


imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
fprintf("Anzahl Bilder: %d\n", length(imageDS.Labels));

rng(7); % Reproduzierbarkeit
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');
fprintf("Trainingsmenge Anzahl Elemente: %d  Test(Validierungs)menge: %d\n", length(trainingImageDS.Labels), length(validationImageDS.Labels));

% Load Pretrained Network

net = alexnet;

% Replace Final 3 Layers, Set the final fully connected layer to have 
%   the same size as the number of classes in the new data set 
%   (5, in this example). To learn faster in the new layers than 
%   in the transferred layers, increase the learning rate factors 
%   of the fully connected layer.

layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingImageDS.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',25,...
        'BiasLearnRateFactor',25)
    softmaxLayer
    classificationLayer];

% Train Network, Specify the training options, including mini-batch size 
%    and validation data. Set InitialLearnRate to a small value to 
%    slow down learning in the transferred layers. In the previous 
%    step, you increased the learning rate factors for the fully 
%    connected layer to speed up learning in the new final layers. 
%    This combination of learning rate settings results in fast learning 
%    only in the new layers and slower learning in the other layers.

options = trainingOptions('sgdm',...
    'MiniBatchSize',128, ...
    'MaxEpochs',300, ...
    'InitialLearnRate',0.0005, ...           % sehr klein -> untere(alte) Layer werden kaum gelernt
    'ValidationData',validationImageDS, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(trainingImageDS,layers,options);

% besser waere ein abschlissendes Testen mit´neuen´ Daten
%  und nicht mit den Validierungsdaten 
YPred = classify(netTransfer, validationImageDS);
accuracy = mean(YPred == validationImageDS.Labels)

%Speichern des trainierten Netzes
save netAlexClassification.mat netTransfer;