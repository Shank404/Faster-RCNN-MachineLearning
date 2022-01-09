% Gruppe 2
% Fabian Beckdorf - 690047
% Jacob Prütz - 690043
% Ali Reza Teimoury - 690065
% Julian Müller - 690018
% Michael Sievers - 690593
% Nico Isheim - 690222
%------------------------------------------------------------------------%
%                    TESTING-REGION DETECTION
%------------------------------------------------------------------------%
% Mithilfe dieses Skriptes wird das ResNet50 getestet.
% Hierbei nutzen wir den uns zur Verfügung gestellten Datensatz, der
% Straßenverkehrsschilder (Testmenge). Im Anschluss werden die erkannten
% Schilder im Ordner "SignsFound" hinterlegt zur weiteren Verarbeitung gesichert.
%------------------------------------------------------------------------%

clear
close all

% ----- Hinzufügen der Arbeitspfade ----- %
addpath Funktionen;

% ----- Laden des belernten Netzes ----- %
load 'Neuronale_Netze/netDetectorResNet50.mat' detector;
inputSize = [448 448 3];

% ----- Laden und randomisieren der Bild-Daten ----- %
[trainingDataDS,validationDataDS,testDataDS,testDataTbl] = LoadAndRandomizeData(inputSize);

% ----- Anpassen des Test-Daten-Datastore ----- %
testDataDS = transform(testDataDS,@(data)preprocessData(data,inputSize));

% ----- Testet das Neuronale Netz mit den Testbildern ----- %
detectionResults = detect(detector,testDataDS,'MinibatchSize',1); 

% ----- Auswertung des Detektors mithilfe der durschnittlichen Präzision ----- %
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testDataDS);

% ----- Ausgabe des Ergebnis Plots ----- %
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
pause(4) % Kurze Pause, damit der Graph angezeigt werden kann

% ----- Speichern der erkannten Schilder ----- %
disp('Start storing images');
basePath = 'SignsFound';
if ~exist(basePath,'dir')
    mkdir(basePath);
end
for i = 1:length(testDataTbl.imageFilename)-1
    img = imread(testDataTbl.imageFilename{i});
    [bboxes,scores] = detect(detector,img);
    rowBBoxes = size(bboxes);
    for j = 1:rowBBoxes(1)
        signImg = imcrop(img, [ bboxes(1*j) bboxes(2*j) bboxes(3*j) bboxes(4*j) ] );
        if ~isempty(signImg)
            imwrite(signImg, fullfile(basePath, strcat(num2str(i),'_',num2str(j),'_', replace(replace(testDataTbl.imageFilename{i},'Pictures_1024_768',''),'\','')) ));
        end
    end
end
