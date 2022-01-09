% Test Alex Netz
close all
clear

% Variablen zur Verteilung der Daten
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten

% Einlesen der erkannten Schilder
imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');

% laden des vortrainierten AlexNetzes
load 'Neuronale Netze\netAlexClassification.mat' netTransfer;

% ----- Anwendung des trainierten Netzwerkes: Gegen den Gesamtdatensatz testen                          
predictedLabels = classify(netTransfer, imageDS);
accuracy = mean(predictedLabels == imageDS.Labels);

% ein Bild entnehmen
[T, info] = read(imageDS);
str = cellstr(info.Label);
image(T);
classify(netTransfer, T);