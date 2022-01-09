function imresized = funcResizeImage(im, out_width, out_height, figureNr, keyStop)
% resized ein Image auf die uebergebenen Werte, falls das AspectRatio nicht
% stimmt werden jeweils aussen schwarze Balken eingefuegt.
% Anzahl der Bildebenen wird nicht angepasst.
%
% formt input/output image: RGB mit uint8 (Pixelwerte im Bereich 0-255)
%
% 15.12.2019   tas
% 25.12.2019   tas   uint8 Behandlung verbessert
%

out_aspectratio = out_width / out_height;

% figure();  imshow(im); title('input')

maxVal = max(max(im));
if maxVal < 2
    fprintf(" Achtung: Aufloesung scheint nicht korretkt zu sein !!!  ")
end

in_width = size(im,2);
in_height = size(im,1);
in_colors = size(im,3);

in_aspectratio = in_width / in_height;
fprintf(" %d*%d*%d  ", in_width, in_height, in_colors)

if size(im) == [out_height out_width 3]
    imresized = im;
    fprintf(" image ist schon ok\n");
    return
end

% expand
if in_aspectratio > out_aspectratio
    % dann hoeher machen, auf das out_aspectratio umformen
    newheight = ceil(in_width/out_aspectratio);
    newwidth = in_width;
    offset = floor((newheight-in_height)/2);
    
    im_expanded = uint8(zeros(newheight, newwidth,3));
    im_expanded(1+offset:in_height+offset, 1:in_width, :) = im;
    % figure();  imshow(im_expanded); title('im_expanded')
    fprintf ("AspectRatio nicht ok ");
elseif in_aspectratio < out_aspectratio
    % dann breiter machen
    newheight = in_height;
    newwidth = ceil(in_height*out_aspectratio);
    offset = floor((newwidth-in_width)/2);
    
    im_expanded = uint8(zeros(newheight,newwidth,3));
    im_expanded(1:in_height,1+offset:in_width+offset,:) = im;
    % figure();  imshow(im_expanded); title('im_expanded')
    fprintf ("AspectRatio nicht ok ");
else
    im_expanded = im;
    % fprintf ("AspectRatio ok ");
end
fprintf("\n")

imresized = imresize(im_expanded, [out_height out_width], 'bilinear');

if  figureNr>0
    figure(figureNr)
    imshow(imresized)
end

if keyStop
    x = input('weiter: Return druecken');
end

end

