function [IoU, acc, se, sp, pr, MedNR] = EvaluIndictor(groundtruth, detecResoult, threshold)
%%���������ڼ���ͼ����ǿ/Ŀ������������۲���
%���룺
%       groundtruth    Ŀ����ͼƬ��0,1��ֵͼ��Ŀ������Ϊ1������Ϊ0
%       detecResoult    Ŀ����������ά����ȡֵ��ΧΪ0-1
%       threshold         һϵ�еķָ���ֵ��һά���飬��Χ����Ϊ0-1
%�����
%       IoU = ������
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
%       1.�������õ�gpu���м���
%       2.Ŀ��������۲����������ڵ�ǰ�⼸�֣���ɸ�����Ҫ��������
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