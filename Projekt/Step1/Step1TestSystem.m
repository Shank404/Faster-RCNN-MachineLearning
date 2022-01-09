% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%  TRAINING & TESTING - ResNet50 (Region Detection) + CNN (Classification)
%------------------------------------------------------------------------%
% Hier befindet sich das hintereinander geschaltete Skript.
% Nach dem Start wird zuerst das ResNet50 (Region Detection) trainiert und getestet,
% die erkannten Datenpunkte werden dann gesammelt und an das CNN (Classification)
% übergeben. Doch zuvor wird das CNN noch trainiert, und im Anschluss mit
% den übergebenen Datenpunkten getestet.
%------------------------------------------------------------------------%

clear;
close all;

% ----- Hinzufügen der Arbeitspfade ----- %
addpath Step1;

% ----- Training des ResNet50 ----- % 
Step1TrainRegionDetection
%pause(3)    % Puffer zum Speichern der Datei

% ----- Testen des des ResNet50 ----- %
if exist('Neuronale Netze/netDetectorResNet50.mat','file')
   Step1TestRegionDetection
else
    disp('Error: Das Neuronale Netz ''netDetectorResNet50.mat'' wurde nicht gefunden');
    return
end

% ----- Training des CNN ----- % 
Step1TrainClassification
pause(3)    % Puffer zum Speichern der Datei

run('Funktionen\resizeImages.m')

% ----- Testen des CNN ----- % 
if (exist('SignsFound','dir') && exist('Neuronale Netze/netClassification.mat','file'))
   Step1TestClassification
else
    disp('Error: Das Neuronale Netz ''netClassification.mat'' wurde nicht gefunden');
    return
end
