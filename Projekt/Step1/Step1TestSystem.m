% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%  	 TESTING - ResNet50 (Region Detection) + CNN (Classification)
%------------------------------------------------------------------------%
% Hier befindet sich das hintereinander geschaltete Skript.
% Nach dem Start wird zuerst das ResNet50 (Region Detection) getestet,
% die erkannten Datenpunkte werden dann gespeichert.
% Im Anschluss wird das trainierte CNN (Classification) ausgeführt um die 
% Schilder zu klassifizieren.
%------------------------------------------------------------------------%
% Der Pfad zum DataStore für die Region Detection ist im Skript
% LoadAndRandomizeData (Line 3) festgelegt.
% Der Pfad zum DataStore für die Classification ist im Skript
% Step1TestRegionDetection (Line 23) festgelegt.

clear;
close all;

% ----- Hinzufügen der Arbeitspfade ----- %
addpath Step1;
addpath Neuronale_Netze;

% ----- Testen des ResNet50 ----- %
if exist('Neuronale_Netze/netDetectorResNet50.mat','file')
    Step1TestRegionDetection
else
    disp('Error: Das Neuronale Netz ''netDetectorResNet50.mat'' wurde nicht gefunden');
    return
end

% ----- Testen des CNN ----- %

if (exist('Neuronale_Netze/netClassification.mat','file'))
   Step1TestClassification
else
    disp('Error: Das Neuronale Netz ''netClassification.mat'' wurde nicht gefunden');
    return
end
