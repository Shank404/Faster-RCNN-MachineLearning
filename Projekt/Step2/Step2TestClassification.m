% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%                    TESTING-CLASSIFICATION
%------------------------------------------------------------------------%
% Mithilfe dieses Skriptes wird das AlexNet getestet.
% Hierbei nutzen wir den uns zur Verfügung gestellten Datensatz, der
% Straßenverkehrsschilder (Testmenge). Im Anschluss wird bespielhaft
% ein Datenpunkt mit BoundingBox, und der zutreffenden Wahrscheinlichkeit ausgegeben.
%------------------------------------------------------------------------%

close all
clear

% Pfad zum DataStore
pathDS = 'SignsCutted';
% ----- Variablen zur Verteilung der Daten ----- %
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten

% ----- Laden der Testdaten in einen Datastore ----- %
imageDS = imageDatastore(pathDS,'IncludeSubfolders',true,'LabelSource','foldernames');

% ----- Laden des trainierten Netzes----- %
load 'Neuronale_Netze\netAlexClassification.mat' netTransfer;

% ----- Anwendung des trainierten Netzes ----- %                         
predictedLabels = classify(netTransfer, imageDS);

% ----- Ausgabe der ermittelten Präzision ----- %   
accuracy = mean(predictedLabels == imageDS.Labels)

% ----- Beispielhafte Ausgabe eines Datenpunktes inkl. BoundingBox ----- %
[T, info] = read(imageDS);
str = cellstr(info.Label);
image(T);
classify(netTransfer, T);