% Test CNN Netz
close all
clear

% Variablen zur Verteilung der Daten
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten

% Einlesen der erkannten Schilder
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore

% 50% der Bilder zum Trainieren, 10% zum Validieren, 40% zum Testen
[trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');

% laden des vortrainierten Netzes
load 'Neuronale Netze/netClassification.mat' net;

predictedLabels = classify(net, validationImageDS);
accuracy = mean(predictedLabels == validationImageDS.Labels);

% ein Bild entnehmen
[T, info] = read(testImageDS);
str = cellstr(info.Label);
image(T);
classify(net, T);

% ----- Anwendung des trainierten Netzwerkes: Gegen den Gesamtdatensatz testen                          
predictedLabels = classify(net, testImageDS);
accuracy = mean(predictedLabels == testImageDS.Labels);