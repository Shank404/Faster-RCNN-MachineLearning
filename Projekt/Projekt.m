close all
clear
doTraining = true; %false ;

% ----- Laden der Bild-Daten ----- %
imageDS = imageDatastore('Pictures_1024_768',...
        "IncludeSubfolders",true,...
        "LabelSource","foldernames");
dataVec = load('signDatasetGroundTruth.mat');  % 
signDatasetTbl = dataVec.signDataset;  % 1125*2 table

% ----- Aufteilung der Daten => 60% Training, 10% Validierung, 30% Testen ----- %  
rng(0)  

shuffledIndicesVec = randperm(height(signDatasetTbl));
idxVec = floor(0.6 * height(signDatasetTbl));

% TRAININGS-Daten => 60%
trainingIdxVec = 1:idxVec;
trainingDataTbl = signDatasetTbl(shuffledIndicesVec(trainingIdxVec),:);

% VALIDIERUNGS-Daten => 10%
validationIdxVec = idxVec+1 : idxVec + 1 + floor(0.1 * length(shuffledIndicesVec) );
validationDataTbl = signDatasetTbl(shuffledIndicesVec(validationIdxVec),:);

% TEST-Daten => 30%
testIdxVec = validationIdxVec(end)+1 : length(shuffledIndicesVec);
testDataTbl = signDatasetTbl(shuffledIndicesVec(testIdxVec),:);

% ----- Erzeugen der einzelnen Datastores, welche dann für das Training, die Validierung und zum Testen verwendet werden. ----- %
imdsTrainDS = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrainDS = boxLabelDatastore(trainingDataTbl(:,'sign'));

imdsValidationDS = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidationDS = boxLabelDatastore(validationDataTbl(:,'sign'));

imdsTestDS = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTestDS = boxLabelDatastore(testDataTbl(:,'sign'));

% ----- Kombination der Image- und LabelBox-Datastores ----- %
trainingDataDS = combine(imdsTrainDS,bldsTrainDS); % erzeugt 'CombinedDatastore
validationDataDS = combine(imdsValidationDS,bldsValidationDS);
testDataDS = combine(imdsTestDS,bldsTestDS);

% ----- Ausgabe eines Bildes mit LabelBox ----- %
data = read(trainingDataDS); %EVENTUELL ZUWEISEN
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
%annotatedImage = imresize(annotatedImage,1);  % nur fuer Darstellung vergrößern
figure
imshow(annotatedImage)

% ----- Erstellung des Faster Regionbased Convolutional Neuronal Network ----- %
inputSize =  [448 448 3]; %[384 512 3]; % [768 1024 3];

if doTraining
    preprocessedTrainingData = transform(trainingDataDS, @(data)preprocessData(data,inputSize));
    % Bilder werden auf die input Size skalliert

% Achtung: dieser DS wird nur zur Ermittlung der anchorBoxes verwendet

% % display one of the training images and box labels.
% while 1==0 %hasdata(preprocessedTrainingData)
%     data = read(preprocessedTrainingData);
%     I = data{1};
%     bbox = data{2};
%     annotatedImage = insertShape(I,'Rectangle',bbox);
%     annotatedImage = imresize(annotatedImage,4);  % nur fuer Darstellung
%     figure(1)
%     imshow(annotatedImage)
%     pause(0.100)
% end

    % ----- Ermittlung der Anchor-boxes ----- %
    numAnchors = 3;
    anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

    % ----- und das feature CNN ----- %
    featureExtractionNetwork = resnet50;
    featureLayer = 'activation_40_relu';
    numClasses = width(signDatasetTbl)-1;    % also hier: 1, es sollen nur Schilder erkannt werden
    lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

    % ----- Netzwerk ansehen ----- %
    % analyzeNetwork(lgraph) 

    % ----- Augmentierung der Daten ----- %
    augmentedTrainingData = transform(trainingDataDS,@augmentData); %u.a. horizontales spiegeln (siehe Funkton unten). Das ist vielleicht keine gute Idee !!
    trainingDataDS = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
    validationDataDS = transform(validationDataDS,@(data)preprocessData(data,inputSize));
    
    options = trainingOptions('sgdm',...
        'MaxEpochs',12,...  % hier reichen wahrscheinlich auch 5
        'MiniBatchSize',1,...
        'InitialLearnRate',1e-3,...
        'CheckpointPath',tempdir,...
        'ValidationData',validationDataDS);

    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    [detector, info] = trainFasterRCNNObjectDetector(trainingDataDS,lgraph,options); %, ...
        %'NegativeOverlapRange',[0 0.3], ...
        %'PositiveOverlapRange',[0.6 1]);
    save netDetectorResNet50.mat detector;
else
    % Load pretrained detector for the example.
    load netDetectorResNet50.mat detector;
end

% ----- quick check/test ----- %
showIndx = floor(rand()*length(testDataTbl.imageFilename)) % Für zufälliges Bild
I = imread(testDataTbl.imageFilename{12});   %I = imread(testDataTbl.imageFilename{3});
%I = imresize(I,inputSize(1:2));    % Nicht resizen, sonst passen die Koordinaten des Rechtecks nicht mehr
[bboxes,scores] = detect(detector,I);

% ----- Ausgabe der Ergebnisse ----- %
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% ----- Testen des Neuronalen Netzes ----- %
testDataDS = transform(testDataDS,@(data)preprocessData(data,inputSize));

% ----- Teste das Neuronale Netz mit den Testbildern ----- %
detectionResults = detect(detector,testDataDS,'MinibatchSize',1); 

% ----- Auswertung des Detektors mithilfe der durschnittlichen Präzision ----- %
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testDataDS);
% The precision/recall (PR) curve highlights how precise a detector is at varying levels of recall. The ideal precision is 1 at all recall levels. The use of more data can help improve the average precision but might require more training time. Plot the PR curve.

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
%   Je größer das Integral unterm Graph, desto besser ist das Netz

% ----- Speichern der erkannten Schilder ----- %
disp('Start storing images');
basePath = 'SignsFound';
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

% ----- Hilfsfunktionen ----- %

function data = augmentData(data)
% Randomly flip images and bounding boxes horizontally.
tform = randomAffine2d('XReflection',true);
sz = size(data{1});
rout = affineOutputView(sz,tform);
data{1} = imwarp(data{1},tform,'OutputView',rout);

% Sanitize box data, if needed.
% data{2} = helperSanitizeBoxes(data{2}, sz);

% Warp boxes.
data{2} = bboxwarp(data{2},tform,rout);
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.
sz = size(data{1},[1 2]);
scale = targetSize(1:2)./sz;
data{1} = imresize(data{1},targetSize(1:2));

% Sanitize box data, if needed.
% data{2} = helperSanitizeBoxes(data{2}, sz);

% Resize boxes.
data{2} = bboxresize(data{2},scale);
end


