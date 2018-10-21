function [auc, IoU, acc, se, sp, pr, MedNR]=newroc(groundtruth, detecResoult, thresh)
%% 本函数用于计算目标检测中的评价指标并绘制ROC曲线,依据是像素数，即面积。
%%函数说明
%输入：
%       groundtruth    目标标记图片，0,1二值图，目标像素为1，背景为0
%       detecResoult    目标检测结果，二维矩阵，取值范围为0-1
%       threshold         一系列的分割阈值，一维数组，取值范围必须为0-1
%输出：
%       auc = The area under the curve
%       IoU = 交并比
%       acc = accuracy
%       se = sensitivity = recall
%       sp = specificity
%       pr = precision
%       f1 = F-measure
%       MedNR = median of normalized responses across the landmark positions
%example:  
%       threshold=0:0.01:1;
%       groundtruth = imread('manual.tif');
%       groundtruth = double(groundtruth)/255;
%       detecResoult = imread('detec.png');
%       detecResoult = double(detecResoult)/255;
%       [auc, IoU, acc, se, sp, pr, MedNR] = newroc(groundtruth, detecResoult, threshold)
%Note:
%       1.本函数用到gpu进行加速
%       2.目标检测的评价参数不局限于当前这几种，你可根据需要自行增减
%Reference:
%       Kumar R, Indrayan A. Receiver operating characteristic (ROC) curve for medical researchers[J]. Indian pediatrics, 2011, 48(4): 277-287.
%
%Author: Liao.L    Liu yuhan
%IDIP.UESTC
%date: 2018-9-19

%%
Testresult = cell(length(thresh)+1,1);
Testresult{1} = ones(size(detecResoult)); 
for j = 1:length(thresh)
    Testresult{j+1} = im2bw(detecResoult,thresh(j));
end
n=length(Testresult);
groundtruth = gpuArray(groundtruth);
atp=zeros(1,n,'gpuArray');
afp=zeros(1,n,'gpuArray');
atn=zeros(1,n,'gpuArray');
afn=zeros(1,n,'gpuArray');
for i=1:n
testresult=Testresult{i};
testresult = gpuArray(testresult);
ngroundtruth=~groundtruth;
ntestresult=~testresult;
TP = sum(sum(groundtruth.*testresult));
FP = sum(sum(testresult-groundtruth.*testresult));
TN = sum(sum(ngroundtruth.*ntestresult));
FN = sum(sum(groundtruth-groundtruth.*testresult));

atp(i)=TP;
afp(i)=FP;
atn(i)=TN;
afn(i)=FN;
end
TPR = (atp)./(atp + afn);
FPR = afp./(afp+atn);
FPR = gather(FPR);
TPR = gather(TPR);
%plot ROC curve
figure,plot(FPR,TPR)
xlabel('FPR'),ylabel('TPR'),title('ROC curve')
%calculation AUC value
auc = CalAUC(FPR,TPR);
%calculation the best threhold according to ROC curve,which means the threhold that
%locals at upper left corner in ROC curve
%根据RIOC曲线，计算最优阈值，所用计算方法是Youden指数
RightIndex = TPR+(1-FPR)-1;
[~,index]=max(RightIndex);
thresholds_val = thresh(index(1)-1);
%
groundtruth = gather(groundtruth);
[IoU, acc, se, sp, pr, MedNR] = EvaluIndictor(groundtruth, detecResoult, thresholds_val);

    