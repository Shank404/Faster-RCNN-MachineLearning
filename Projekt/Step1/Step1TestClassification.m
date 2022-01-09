% Test CNN Netz
close all
clear

% Variablen zur Verteilung der Daten
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten

useFoundSigns = true;
inputSize = [227, 227, 3];                          %Skalierung für das Netzwerk

if ~useFoundSigns
    % Einlesen der Schilder aus dem Datastore
    imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
    % 50% der Bilder zum Trainieren, 10% zum Validieren, 40% zum Testen
    [trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');

else
    % Einlesen der erkannten Schilder
    testImageDS = imageDatastore('SignsFound'); %,'IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
end

% laden des vortrainierten Netzes
load 'Neuronale Netze/netClassification.mat' net;

predictedLabels = classify(net, testImageDS);
if useFoundSigns
    imageDS.Labels = predictedLabels;
else
    accuracy = mean(predictedLabels == imageDS.Labels)
end

% ein Bild entnehmen
[T, info] = read(imageDS);
str = cellstr(info.Label);
image(T);
classify(net, T);

% ----- Anwendung des trainierten Netzwerkes: Gegen den Gesamtdatensatz testen
if ~useFoundSigns
    predictedLabels = classify(net, imageDS);
    accuracy = mean(predictedLabels == imageDS.Labels)
end
