clear all; close all; clc;

%% Input parameters
NeedleD = 0.211; %508 Needle Diameter in centimeters
nPoints = 5; %Number of pixels to use for determining Needle Diameter
resolutionLimit = 3; % pixels
jumpLimit = 20; % pixels
folder = '/Users/ypm/Desktop/mat_drop/';
fileTag = '*.mp4';
doQC = 0;
firstFrame = 1;
lastFrame = 0;
manualCrop = 0;
rectMaster = 0;%[50  185  253  253];
visualize = 0; % 0 for false, 1 for true
rotate = 1*-90; % angle in degrees
fpsR = 3587; %Need to record when using new camera
row = 2; 

%% Read video & calculation

%Parse inputs
fileList = dir([folder,fileTag]);
savedList = dir([folder,'*.mat']);
N = length(fileList);

for j = 1:N % Loop through all files
        
    % Check to see if video is already processed
    vidName = fileList(j).name;
    saveName = vidName; % Use the same file name as the figure
    saveName(end-2:end) = 'mat';
    fullSavePath = [folder saveName];
    if isfile(fullSavePath)
        disp(saveName);
        continue
    else
        disp(vidName)
    end
    
    % Get video file and parameters
    vidObj = VideoReader([folder,vidName]);
    nFrames = vidObj.NumberOfFrames;
    temp = split(vidName,'fps=');
    temp = temp{end};
    fps = str2double(temp(1:end-6));
    %assert(~isnan(fps));
        
    
    % Parse video parameters
    finish = nFrames - lastFrame;
    rNeck1 = zeros(1,nFrames);%Create array of all zeros
    rNeck2 = zeros(1,nFrames);%Create array of all zeros
    
    % Get the needle radius
    num = round(.05*nFrames);
    if num < 1; num = 1; end
    frame = imrotate(read(vidObj,num),rotate);
    
    figure()
    imshow(frame)
    title('Crop just a small section of the capillary');
    if manualCrop
        [frame,rect] = imcrop(gcf);
    else
        pause(0.5)
        height = size(frame,1);
        width = size(frame,2);
        rect = [5,5,width-5,10];
        frame = imcrop(frame,rect);
    end
        
    MONO = rgb2gray(frame);
    level = graythresh(MONO); %Manually set reference
    BW = imcomplement(imbinarize(MONO,level));%Reverse Black and White
    BW_filled = imfill(BW,'holes');%Fill image regions and holes
    col = find(BW_filled(row,:), 1);
    contour = bwtraceboundary(BW_filled,[row, col],'S', 8, inf,...
        'counterclockwise');
    x = contour(:,2); 
    y = contour(:,1); 
    D0 = max(x) - min(x);
    R0 = D0/2;
    imshow(frame)
    hold on
    plot(x,y,'b.')
    pause(.5)
    close(gcf)
    
    % Set the cropping for isolating the neck region
    if manualCrop
        num = round(.5*nFrames);
        frame = imrotate(read(vidObj,num),rotate);
        figure()
        imshow(frame)
        title('Crop the section where the neck will be.');
        [~,rect] = imcrop(gcf);
        close(gcf)
    else
        rect = [10,40,width-10,height-40];
    end
    
    % Loop through frames
    for i = firstFrame:finish
        % Get one frame and rotate
        frame = imrotate(read(vidObj,i), rotate);
            
        %crop and covert to monochromatic 
        frame = imcrop(frame,rect);
        MONO = rgb2gray(frame);
        
        % Find contour
        temp = double(max(MONO(:)))/255;
        level = (0.5*temp + 1.5*graythresh(MONO))/2; %Manually set reference
        BW = imcomplement(imbinarize(MONO,level));%Reverse Black and White
        BW_filled = imfill(BW,'holes');%Fill image regions and holes
        col = find(BW_filled(row,:), 1);
        contour = bwtraceboundary(BW_filled,[row, col],'S', 8, inf,...
            'counterclockwise');
        
        % Select portions of the contour
        x = contour(:,2); 
        y = contour(:,1);
        [height,width] = size(BW);
        trimPixels = 15;
        temp = find(y>trimPixels); 
        if isempty(temp)
            continue
        end
        ind1 = temp(1); ind4 = temp(end);
        temp = find(y>=(height-trimPixels)); 
        if isempty(temp)
            temp = find(y==max(y));
        end
        ind2 = temp(1); ind3 = temp(end);
        x1 = x(ind1:ind2);
        y1 = y(ind1:ind2);
        x2 = x(ind3:ind4);
        y2 = y(ind3:ind4);
        
        % Find the neck radius in two ways
        [xLeft,leftInd] = max(x1);
        yLeft = y1(leftInd);
        rightInd = find(y2==yLeft,1);
        xRight = x2(rightInd);
        yRight = y2(rightInd);
        rNeck1(i) = (xRight - xLeft)/2;
        x2Match = 0*x1+width;
        for j = 1:length(x1)
            if (y1(j) < min(y2)) || (y1(j) > max(y2))
                continue
            end            
            x2MatchInd = find(y2==y1(j),1);
            x2Match(j) = x2(x2MatchInd);
        end
        [rNeck2(i),minInd] = min(x2Match-x1);
        rNeck2(i) = rNeck2(i)/2;
        xL2 = x1(minInd);
        yL2 = y1(minInd);
        
        % Visualize analysis
        if visualize
            figure(123)
            imshow(frame);
            hold on;
            yNeedle1 = y(1:nPoints); 
            yNeedle2 = y(end-nPoints+1:end);
            plot(x,y,'b.')
            plot(x1,y1,'g.',x2,y2,'m.');
            plot([xLeft,xRight],[yLeft,yRight],'ko');
            plot(xL2,yL2,'ro')
            pause(0.001)
            if mod(i,100)==0; cla; disp(i); end
        end

    end

    % Calculate time from frame rate and plot
    tStep = 1/duration; %duration of each frame
    t1 = (0:length(rNeck1)-1)*tStep*1000;
    t2 = (0:length(rNeck2)-1)*tStep*1000;
    
    W1 = rNeck1/R0;
    W2 = rNeck2/R0;
    
    % Remove low resolution data
    cutoff = resolutionLimit/R0/2;
    t1 = t1(W1>=cutoff);
    rNeck1 = rNeck1(W1>=cutoff);
    t2 = t2(W2>=cutoff);
    rNeck2 = rNeck2(W2>=cutoff);
    W1 = W1(W1 >= cutoff);
    W2 = W2(W2 >= cutoff);


    figure()
    semilogy(t1,W1,'ko',t2,W2,'bx');
    xlabel('t [ms]','FontSize',12);
    ylabel('Rneck/R0','FontSize',12);
    title(vidName)
    set(gca,'FontSize',12);
    pause(0.5)
    
    if doQC
        answer = questdlg('Does the data look okay?','QC','Yes','No','No');
    else
        answer = 'Yes';
    end
    
    if strcmp(answer,'Yes')
        save(fullSavePath,'fps','NeedleD','R0','rNeck1','rNeck2','t1',...
            't2')
    % Load the variables from the .mat file
    %data = load('testfile_copy.mat');

    % Determine the maximum number of rows among all variables
    %maxRows = max([numel(data.NeedleD), numel(data.R0), numel(data.rNeck1), numel(data.rNeck2), numel(data.t1), numel(data.t2)]);

    % Pad the variables with NaN values to match the maximum number of rows
    %paddedNeedleD = padarray(data.NeedleD(:), maxRows - numel(data.NeedleD), NaN, 'post');
    %paddedR0 = padarray(data.R0(:), maxRows - numel(data.R0), NaN, 'post');
    %paddedrNeck1 = padarray(data.rNeck1(:), maxRows - numel(data.rNeck1), NaN, 'post');
    %paddedrNeck2 = padarray(data.rNeck2(:), maxRows - numel(data.rNeck2), NaN, 'post');
    %paddedt1 = padarray(data.t1(:), maxRows - numel(data.t1), NaN, 'post');
    %paddedt2 = padarray(data.t2(:), maxRows - numel(data.t2), NaN, 'post');

    % Create a table with the padded variables
    %T = table(paddedNeedleD, paddedR0, paddedrNeck1, paddedrNeck2, paddedt1, paddedt2, 'VariableNames', {'NeedleD', 'R0', 'rNeck1', 'rNeck2', 't1', 't2'});

    % Save the table to a CSV file
    %writetable(T, '0.001 wt.% N12K, trial 3, June 20.csv');
    end

    % Get a list of all .mat files in the folder
    matFiles = dir('*.mat');

% Iterate over each .mat file
    for i = 1:numel(matFiles)
    % Load the variables from the current .mat file
        data = load(matFiles(i).name);

    % Determine the maximum number of rows among all variables
        maxRows = max([numel(data.NeedleD), numel(data.R0), numel(data.rNeck1), numel(data.rNeck2), numel(data.t1), numel(data.t2)]);

    % Pad the variables with NaN values to match the maximum number of rows
        paddedNeedleD = padarray(data.NeedleD(:), maxRows - numel(data.NeedleD), NaN, 'post');
        paddedR0 = padarray(data.R0(:), maxRows - numel(data.R0), NaN, 'post');
        paddedrNeck1 = padarray(data.rNeck1(:), maxRows - numel(data.rNeck1), NaN, 'post');
        paddedrNeck2 = padarray(data.rNeck2(:), maxRows - numel(data.rNeck2), NaN, 'post');
        paddedt1 = padarray(data.t1(:), maxRows - numel(data.t1), NaN, 'post');
        paddedt2 = padarray(data.t2(:), maxRows - numel(data.t2), NaN, 'post');

    % Create a table with the padded variables
        T = table(paddedNeedleD, paddedR0, paddedrNeck1, paddedrNeck2, paddedt1, paddedt2, 'VariableNames', {'NeedleD', 'R0', 'rNeck1', 'rNeck2', 't1', 't2'});

    % Generate the CSV file name by replacing the .mat extension with .csv
        csvFileName = strrep(matFiles(i).name, '.mat', '.csv');

    % Save the table to the CSV file
        writetable(T, csvFileName);
    end

end