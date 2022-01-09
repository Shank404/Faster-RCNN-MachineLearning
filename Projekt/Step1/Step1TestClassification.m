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
% Mithilfe dieses Skriptes wird das CNN getestet.
% Hierbei nutzen wir den uns zur Verfügung gestellten Datensatz, der
% Straßenverkehrsschilder (Testmenge). Im Anschluss wird bespielhaft
% ein Datenpunkt mit BoundingBox, und der zutreffenden Wahrscheinlichkeit ausgegeben.
% Sollten die Daten vewendet werden welche vom ResNet50 erkannt wurden,
% muss der boolean useFoundSigns auf true gesetzt werden.
%------------------------------------------------------------------------%

close all
clear

% ----- Nutzung der erkannten Daten des FRCNN ----- %
useFoundSigns = true;

% ----- Variablen zur Verteilung der Daten ----- %
amountTrain = 0.5;                                  %Anzahl der Trainingsdaten
amountVal = 0.1;                                    %Anzahl der Validierungsdaten
amountTest = 0.4;                                   %Anzahl der Testdaten

% ----- Auswahl des Datensatzes ----- %
if ~useFoundSigns
    % Einlesen der Schilder aus dem Datastore
    imageDS = imageDatastore('SignsCutted','IncludeSubfolders',true,'LabelSource','foldernames');  % create DataStore
    [trainingImageDS, validationImageDS, testImageDS] = splitEachLabel(imageDS, amountTrain, amountVal, amountTest,'randomized');

else
    % Einlesen der erkannten Schilder
    testImageDS = imageDatastore('SignsFound');
end

% ----- Laden des trainierten Netzes----- %
load 'Neuronale_Netze/netClassification.mat' net;
inputSize = [227, 227, 3]; %Skalierung für das Netzwerk

% ----- Anwendung des trainierten Netzes ----- %                         
predictedLabels = classify(net, testImageDS);

% ----- Ausgabe der ermittelten Präzision ----- %  
if useFoundSigns
    imageDS.Labels = predictedLabels;
else
    accuracy = mean(predictedLabels == imageDS.Labels)
end

% ----- Beispielhafte Ausgabe eines Datenpunktes inkl. BoundingBox ----- %
[T, info] = read(imageDS);
str = cellstr(info.Label);
image(T);
classify(net, T);

% ----- Anwendung des trainierten Netzwerkes: Gegen den Gesamtdatensatz testen ----- %
if ~useFoundSigns
    predictedLabels = classify(net, imageDS);
    accuracy = mean(predictedLabels == imageDS.Labels)
end
