%
%   groundTruthTableFromLabelPics
%
%   erstellt us einem Picture-Datastore und einem Label-Datastore
%   eine Groundtruth-Tabelle, wie sie z.B. in FasterRCNN.m
%   benoetigt wird
%
%   Anpassungen:
%   - Datastore pathes
%   - Name Groundtruthtabelle (am Ende)
%   - u.U. kann der pictureCount und der labelCount erniedrigt werden
%
% 07.12.2021  tas
% 14.12.2021  tas
%

clear all; close all;

%%%%%%%%%%%%%% Parameter %%%%%%%%%%%%%%%%%
% dataStorePicturePath = 'C:\Users\tas\Desktop\211202 Mat DeepLearning Examples Razor\eTrafficSignsData\SS21TrafficSigns224_224_3\Pictures\30GBS'
% dataStoreLabelPath = 'C:\Users\tas\Desktop\211202 Mat DeepLearning Examples Razor\eTrafficSignsData\SS21TrafficSigns224_224_3\Labels\30GBS'
dataStorePicturePath ="Pictures_1024_768"
dataStoreLabelPath ="Labels_1024_768"
labelDS = imageDatastore(dataStoreLabelPath, 'IncludeSubfolders', true);
pictureDS = imageDatastore(dataStorePicturePath, 'IncludeSubfolders', true);

labelCount = numel(labelDS.Files)
pictureCount = numel(pictureDS.Files)

% nur fuer Tests:
% pictureCount = 1000
% labelCount = 1000

if labelCount ~= pictureCount
    fprintf("!!!! Error: Die Anzahl der Bilder und Anzahl der LabelPicture sind ungleich -> Abbruch");
    return
end

fprintf("-----------------------------------------------------------\n");
fprintf("Picture-Verzeichnis: %s\n", dataStorePicturePath)
fprintf("Anzahl Images: %d\n", pictureCount)
fprintf("Label-Verzeichnis: %s\n", dataStoreLabelPath)
fprintf("Anzahl Images: %d\n", labelCount)
fprintf("-----------------------------------------------------------\n");

% table anlegen
sz = [pictureCount 2];
varTypes = ["cellstr","cell"];
varNames = ["imageFilename","sign"];
signDataset = table('Size',sz,'VariableTypes',varTypes,'VariableNames',varNames);

rng(0)
shuffledIndices = randperm(pictureCount);

% los gehts
for i = 1:pictureCount
    
    shuffeldIndex = shuffledIndices(i);
    [imPic imPic_INFO]= readimage(pictureDS, shuffeldIndex);
    [im_path imPic_name im_ext]=fileparts(imPic_INFO.Filename);
    
    [imLabel imLabel_INFO]= readimage(labelDS, shuffeldIndex);
    [im_path imLabel_name im_ext]=fileparts(imLabel_INFO.Filename);
    
    % fprintf("picture: %s label: %s\n", imPic_name, imLabel_name);
    
    if ~strcmp(imPic_name, imLabel_name)
        fprintf("!!!! Error: zum Picture gibt es kein entsprechendes LabelPicture -> Abbruch");
        imPic_name
        imLabel_name
        return
    end
    
    % LabelRegion aus Image ausschneiden
    bw = imLabel;
    s = regionprops(bw, 'BoundingBox');
    box = cat(1, s.BoundingBox); % structure to matrix
    box = round(box);  
    
    % falls mehrere Marker vorhanden sind, nur einen uebernehmen
    box = box(1,:);
    
    if (numel(box) ~= 4)
        fprintf("Boxkoordinaten nicht ok: %s %s\n", imPic_name, imLabel_name)
        box
    end
    
    a = num2cell(box, 2);
    signDataset(shuffeldIndex,:) = {imPic_INFO.Filename,a};
    
    % display one of the training images and box labels.
    if (i == 4)
        annotatedImage = insertShape(imPic,'Rectangle',box);
        figure
        imshow(annotatedImage)
    end
    
end
signDataset

% save('signDatasetGroundTruth_1024_768_3_shuffeld_1000.mat', 'signDataset');
save('signDatasetGroundTruth.mat', 'signDataset');
