function [auc, IoU, acc, se, sp, pr, MedNR]=newroc(groundtruth, detecResoult, thresh)
%% ���������ڼ���Ŀ�����е�����ָ�겢����ROC����,���������������������
%%����˵��
%���룺
%       groundtruth    Ŀ����ͼƬ��0,1��ֵͼ��Ŀ������Ϊ1������Ϊ0
%       detecResoult    Ŀ����������ά����ȡֵ��ΧΪ0-1
%       threshold         һϵ�еķָ���ֵ��һά���飬ȡֵ��Χ����Ϊ0-1
%�����
%       auc = The area under the curve
%       IoU = ������
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
%       1.�������õ�gpu���м���
%       2.Ŀ��������۲����������ڵ�ǰ�⼸�֣���ɸ�����Ҫ��������
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
%����RIOC���ߣ�����������ֵ�����ü��㷽����Youdenָ��
RightIndex = TPR+(1-FPR)-1;
[~,index]=max(RightIndex);
thresholds_val = thresh(index(1)-1);
%
groundtruth = gather(groundtruth);
[IoU, acc, se, sp, pr, MedNR] = EvaluIndictor(groundtruth, detecResoult, thresholds_val);

    