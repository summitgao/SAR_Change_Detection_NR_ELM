
% The files are the MATLAB source code for the paper:
% Feng Gao,  Junyu Dong, Bo Li, Qizhi Xu, Cui Xie. 
% Change Detection from Synthetic Aperture Radar Images Based on Neighborhood-Based Ratio and Extreme Learning Machine 
% Journal of Applied Remote Sensing. 10(4), 2016.
%
% The demo has not been well organized. Please contact me if you meet any problems.
% 
% Email: gaofeng@ouc.edu.cn
% 
% 注意事项：
%   因为训练时样本是随机生成，样本不同会影响最终的结果。
%   所以每次运行的结果都不会一样。
%   作者会在后期发布的版本中进行修正（固定生成随机数的种子）
%



clear;
clc;
close all;

addpath('./Utils');

PatSize = 5;
k_n = 3;

fprintf(' ... ... read image file ... ... ... ....\n');
im1   = imread('./pic/san_1.bmp');
im2   = imread('./pic/san_2.bmp');
im_gt = imread('./pic/san_gt.bmp');
fprintf(' ... ... read image file finished !!! !!!\n\n');

im1 = double(im1(:,:,1));
im2 = double(im2(:,:,1));
im_gt = double(im_gt(:,:,1));

[ylen, xlen] = size(im1);

% compute the neighborhood-based ratio image
fprintf(' ... .. compute the neighborhood ratio ..\n');
nrmap = nr(im1, im2, k_n);
nrmap = max(nrmap(:))-nrmap;
nrmap = nr_enhance( nrmap );
feat_vec = reshape(nrmap, ylen*xlen, 1);
fprintf(' ... .. compute finished !!! !!! !!! !!!!\n\n');

fprintf(' ... .. clustering for sample selection begin ... ....\n');
im_lab = gao_clustering(feat_vec, ylen, xlen);
fprintf(' ... .. clustering for sample selection finished !!!!!\n\n');

fprintf(' ... ... ... samples initializaton begin ... ... .....\n');
fprintf(' ... ... ... Patch Size : %d pixels ... ....\n', PatSize);


pos_lab = find(im_lab == 1);
neg_lab = find(im_lab == 0);

pos_lab = pos_lab(randperm(numel(pos_lab)));
neg_lab = neg_lab(randperm(numel(neg_lab)));

[ylen, xlen] = size(im1);

mag = (PatSize-1)/2;
imTmp = zeros(ylen+PatSize-1, xlen+PatSize-1);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im1; 
im1 = im2col_general(imTmp, [PatSize, PatSize]);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im2; 
im2 = im2col_general(imTmp, [PatSize, PatSize]);
clear imTmp mag;

% merge samples to im
im1 = mat2imgcell(im1, PatSize, PatSize, 'gray');
im2 = mat2imgcell(im2, PatSize, PatSize, 'gray');
parfor idx = 1 : numel(im1)
    im_tmp = [im1{idx}; im2{idx}];
    im(idx, :) = im_tmp(:);
end
clear im1 im2 idx;

fprintf(' ... ... ... randomly generation samples ... ... .....\n');
PosNum = numel(pos_lab);
NegNum = numel(neg_lab);


% 取出正负样本图像块
PosPat = im(pos_lab(1:PosNum), :);
NegPat = im(neg_lab(1:NegNum), :);
TrnPat = [PosPat; NegPat];
TrnLab = [ones(PosNum, 1); zeros(NegNum, 1)];
trn_data = [TrnLab, TrnPat];
clear PosPat NegPat TraPat TrnLab; 
clear PosNum NegNum;
clear pos_lab neg_lab;

TstLab = ones(size(im,1), 1);
tst_data = [TstLab, im];

fprintf(' ============== Extreme Learning Machine begin ========\n');
%-----------------------------------------------------
trn_accu = elm_train(trn_data, 1, 300, 'sig');
elm_out = elm_predict(tst_data);


elm_out = reshape(elm_out, [ylen, xlen]);
idx = find(im_lab == 1);
elm_out(idx) = 1;
idx = find(im_lab == 0);
elm_out(idx) = 0;
fprintf(' ============== Extreme Learning Machine finished !!!!!\n\n');

clear trn_data tst_data trn_accu;
clear ylen xlen;

[elm_out,num] = bwlabel(~elm_out);
for i = 1:num
    idx = find(elm_out==i);
    if numel(idx) <= 10
        elm_out(idx)=0;
    end
end
elm_out = elm_out>0;
clear i num TrnPat TstLab NumSam;

[FA,MA,OE,CA] = DAcom(im_gt, elm_out);
% Save change detection results
fid = fopen('rec.txt', 'a');
fprintf(fid, 'PatSize = %d\n', PatSize);
fprintf(fid, 'False Alarm   Pixels Number: %d \n', FA);
fprintf(fid, 'Miss Detected Pixels Number: %d \n', MA);
fprintf(fid, 'Overall Error Pixels Number: %d \n', OE);
fprintf(fid, 'PCC :   %f \n\n\n', CA);
fclose(fid);

fprintf(' ===== Written change detection results to Res.txt ====\n\n');









