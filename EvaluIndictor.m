function [IoU, acc, se, sp, pr, MedNR] = EvaluIndictor(groundtruth, detecResoult, threshold)
%%本函数用于计算图像增强/目标检测领域的评价参数
%输入：
%       groundtruth    目标标记图片，0,1二值图，目标像素为1，背景为0
%       detecResoult    目标检测结果，二维矩阵，取值范围为0-1
%       threshold         一系列的分割阈值，一维数组，范围必须为0-1
%输出：
%       IoU = 交并比
%       acc = accuracy
%       se = sensitivity = recall
%       sp = specificity
%       pr = precision
%       f1 = F-measure
%       MedNR = median of normalized responses across the landmark positions
%example:  
%       threshold=0:0.01:1;
%       groundtruth = imread('mannul.tif');
%       groundtruth = double(groundtruth)/255;
%       detecResoult = imread('detec.png');
%       detecResoult = double(detecResoult)/255;
%       [IoU, acc, se, sp, pr, MedNR] = EvaluIndictor(groundtruth, detecResoult, threshold)
%Note:
%       1.本函数用到gpu进行加速
%       2.目标检测的评价参数不局限于当前这几种，你可根据需要自行增减
%Reference:
%       Kumar R, Indrayan A. Receiver operating characteristic (ROC) curve for medical researchers[J]. Indian pediatrics, 2011, 48(4): 277-287.
%
%Author: Liao.L
%IDIP.UESTC
%date: 2018-9-19
%%

landmark = groundtruth.*detecResoult;
landmark = nonzeros(landmark);
MedNR = median(landmark);

testresult = double(im2bw(detecResoult,threshold));
testresult = gpuArray(testresult);
groundtruth = gpuArray(groundtruth);
ngroundtruth=~groundtruth;
ntestresult=~testresult;
tp = sum(sum(groundtruth.*testresult));
fp = sum(sum(testresult-groundtruth.*testresult));
tn = sum(sum(ngroundtruth.*ntestresult));
fn = sum(sum(groundtruth-groundtruth.*testresult));

IoU = tp/(tp + fp + fn);
acc = (tp+tn)/(tp+fn+fp+tn);
se = tp/(tp+fn);
sp = tn/(fp+tn);
pr = tp/(tp+fp);

IoU = gather(IoU);
acc = gather(acc);
se = gather(se);
sp = gather(sp);
pr = gather(pr);
end