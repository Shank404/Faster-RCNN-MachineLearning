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


run Step2TestRegionDetection.m
run Step2TestClassification.m
