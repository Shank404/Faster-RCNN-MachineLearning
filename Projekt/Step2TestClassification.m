
% Test CNN Netz

close all
clear

% Einlesen der erkannten Schilder
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore

% 50% der Bilder zum Trainieren, 10% zum Validieren, 40% zum Testen
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, 0.5, 0.1, 0.4,'randomized');

%laden des vortrainierten Netzes
load netClassification.mat net;

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