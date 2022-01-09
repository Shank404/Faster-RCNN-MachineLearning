% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%  TRAINING & TESTING - ResNet50 (Region Detection) + AlexNet (Classification)
%------------------------------------------------------------------------%
% Hier befindet sich das hintereinander geschaltete Skript.
% Nach dem Start wird zuerst das ResNet50 (Region Detection) trainiert und getestet,
% die erkannten Datenpunkte werden dann gesammelt und an das AlexNet (Classification)
% übergeben. Doch zuvor wird das AlexNet noch trainiert, und im Anschluss mit
% den übergebenen Datenpunkten getestet.
%------------------------------------------------------------------------%

clear;
close all;

% ----- Hinzufügen der Arbeitspfade ----- %
addpath Step2;
addpath "Neuronale Netze";

% ----- Training des ResNet50 ----- % 
Step2TrainRegionDetection
%pause(3)    % Puffer zum Speichern der Datei

% ----- Testen des des ResNet50 ----- %
if exist('Neuronale Netze/netDetectorResNet50.mat','file')
   Step2TestRegionDetection
else
    disp('Error: Das Neuronale Netz ''netDetectorResNet50.mat'' wurde nicht gefunden');
    return
end

% ----- Training des AlexNet ----- % 
Step2TrainClassification
pause(3)    % Puffer zum Speichern der Datei
run('Funktionen\resizeImages.m')

% ----- Testen des AlexNet ----- % 
if (exist('SignsFound','dir') && exist('Neuronale Netze/netClassification.mat','file'))
   Step2TestClassification
else
    disp('Error: Das Neuronale Netz ''netClassification.mat'' wurde nicht gefunden');
    return
end