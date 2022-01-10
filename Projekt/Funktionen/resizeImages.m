%
% resized die Images eines DataStores, speichert das Ergebnis in einen
%   weiteren DataStore
%
% 15.12.2019, 25.12.2019  tas
%

clear; close all;

%%%%%%%%%%%%%% Parameter %%%%%%%%%%%%%%%%%
keyStop = false
path = '..\SignsFound';
% sourceDataStore = strcat(path, 'ImageLabelsTrain240_180')
% targetDataStore = strcat(path, 'ImageLabelsTrain240_176')
sourceDataStore = strcat(path);
targetDataStore = strcat(path);

targetWidth = 227; %240;
targetHeight = 227; %176;
targetColors = 3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sourceImages = imageDatastore(sourceDataStore);
imageCount = numel(sourceImages.Files);
fprintf("-----------------------------------------------------------\n");
fprintf("SourceVerzeichnis: %s\n", sourceDataStore)
fprintf("TargetVerzeichnis: %s\n", targetDataStore)
fprintf("Anzahl Images: %d\n", imageCount)
fprintf("Targets: %d*%d*%d\n", targetWidth, targetHeight, targetColors)
fprintf("-----------------------------------------------------------\n");

for i = 1:imageCount
   [im im_INFO]= readimage(sourceImages, i);
   [im_path im_name im_ext]=fileparts(im_INFO.Filename);
   
   fprintf("%s%s %.0fKB", im_name, im_ext, im_INFO.FileSize/1024)
   figureNr = 3;
   imResized = funcResizeImage(im, targetWidth, targetHeight, figureNr, keyStop);
   
   if ~exist(targetDataStore, 'dir')
       mkdir(targetDataStore)
   end
   filename = char(im_name +""+ im_ext);   %filename = char(im_name + ".png");
   fname=fullfile(targetDataStore, filename);
   imwrite(imResized,fname, upper(replace(im_ext,'.','')) ); %imwrite(imResized,fname, 'PNG');
end