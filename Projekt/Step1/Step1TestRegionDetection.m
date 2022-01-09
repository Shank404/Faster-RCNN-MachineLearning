% ----- Laden des belernten Netzes ----- %
load 'Neuronale Netze/netDetectorResNet50.mat' detector;

% ----- Laden und randomisieren der Bild-Daten ----- %
[trainingDataDS,validationDataDS,testDataDS,testDataTbl] = LoadAndRandomizeData();
inputSize = [448 448 3];    % => nach Rücksprache mit Aschmoneit

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
