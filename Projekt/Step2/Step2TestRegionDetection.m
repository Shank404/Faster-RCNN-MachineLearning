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
