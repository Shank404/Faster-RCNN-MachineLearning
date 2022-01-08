% ----- Laden des belernten Netzes
load netDetectorResNet50.mat detector;

[trainingDataDS,validationDataDS,testDataDS,testDataTbl] = LoadAndRandomizeData();
inputSize = [448 448 3];

% ----- Testen des Neuronalen Netzes
testDataDS = transform(testDataDS,@(data)preprocessData(data,inputSize));

% ----- Teste das Neuronale Netz mit den Testbildern
detectionResults = detect(detector,testDataDS,'MinibatchSize',1); 

% ----- Auswertung des Detektors mithilfe der durschnittlichen Präzision
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testDataDS);
% The precision/recall (PR) curve highlights how precise a detector is at varying levels of recall. The ideal precision is 1 at all recall levels. The use of more data can help improve the average precision but might require more training time. Plot the PR curve.

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))

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
        %figure('Name',testDataTbl.imageFilename{i},'NumberTitle', 'off');
        %imshow(signImg)
        if ~isempty(signImg)
            imwrite(signImg, fullfile(basePath, strcat(num2str(i),'_',num2str(j),'_', replace(replace(testDataTbl.imageFilename{i},'Pictures_1024_768',''),'\',''), '.jpg') ));
        end
    end
end
