% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%     TESTING - ResNet50 (Region Detection) + AlexNet (Classification)
%------------------------------------------------------------------------%
% Hier befindet sich das hintereinander geschaltete Skript.
% Nach dem Start wird zuerst das ResNet50 (Region Detection) getestet,
% die erkannten Datenpunkte werden dann gesammelt und an das AlexNet (Classification)
% übergeben. Dort werden dann die einzelnen Bilder klassifiziert.
%------------------------------------------------------------------------%

clear;
close all;

addpath Step2;
% --- Trainieren des Region Detection Netzwerks (zum Finden der Schilder im
% Bild)
Step2TrainRegionDetection
%pause(3)    % Puffer zum Speichern der Datei

% --- Testen des Region Detection Netzwerks (zum Finden der Schilder im
% Bild)
if exist('Neuronale Netze/netDetectorResNet50.mat','file')
   Step2TestRegionDetection
else
    disp('Error: Das Neuronale Netz ''netDetectorResNet50.mat'' wurde nicht gefunden');
    return
end

% --- Trainieren des Classification Netzwerks (zum Identifizieren der 
% gefundenen Schilder)
Step2TrainClassification
pause(3)    % Puffer zum Speichern der Datei

run('Funktionen\resizeImages.m')
% --- Testen des Classification Netzwerks (zum Finden der Schilder im
% Bild)
if (exist('SignsFound','dir') && exist('Neuronale Netze/netClassification.mat','file'))
   Step2TestClassification
else
    disp('Error: Das Neuronale Netz ''netClassification.mat'' wurde nicht gefunden');
    return
end