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
