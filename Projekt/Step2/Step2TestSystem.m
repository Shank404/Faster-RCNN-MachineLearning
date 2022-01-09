% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%    TESTING - ResNet50 (Region Detection) + AlexNet (Classification)
%------------------------------------------------------------------------%
% Hier befindet sich das hintereinander geschaltete Skript.
% Nach dem Start wird das ResNet50 (Region Detection) getestet,
% die erkannten Datenpunkte werden dann gespeichert
% und mit dem AlexNet (Classification) klassifiziert.
%------------------------------------------------------------------------%

clear;
close all;

% ----- Hinzufügen der Arbeitspfade ----- %
addpath Step2;
addpath 'Neuronale_Netze';

% ----- Testen des ResNet50 ----- %
if exist('Neuronale_Netze/netDetectorResNet50.mat','file')
   Step2TestRegionDetection
   run('Funktionen\resizeImages.m')
else
    disp('Error: Das Neuronale Netz ''netDetectorResNet50.mat'' wurde nicht gefunden');
    return
end

% ----- Testen des AlexNet ----- % 

if (exist('SignsFound','dir') && exist('Neuronale_Netze/netClassification.mat','file'))
   Step2TestClassification
else
    disp('Error: Das Neuronale Netz ''netClassification.mat'' wurde nicht gefunden');
    return
end