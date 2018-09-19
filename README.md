# NewROC
本函数由于图像中目标检测算法的评价，依据的是**ROC曲线** <br>
本函数可以绘制ROC曲线，并计算相应的评价指标，如*auc, sensitivity, specificity* 等 <br>
***
## example:  
>threshold=0:0.01:1;<br>
>groundtruth = imread('mannul.tif');<br>
>groundtruth = double(groundtruth)/255;<br>
>detecResoult = imread('detec.png');<br>
>detecResoult = double(detecResoult)/255;<br>
>[auc, IoU, acc, se, sp, pr, MedNR] = newroc(groundtruth, detecResoult, threshold);
## Note:
       1.本函数用到gpu进行加速
       2.目标检测的评价参数不局限于当前这几种，你可根据需要自行增减
## Reference:
*Kumar R, Indrayan A. Receiver operating characteristic (ROC) curve for medical researchers[J]. Indian pediatrics, 2011, 48(4): 277-287.*

***
```Auther:``` [```LIAO.L```](http://gispalab.uestc.edu.cn/studentNow/849.htm "Markdown") <br>
[```IDIPLAB·UESCT```](http://gispalab.uestc.edu.cn "Markdown")
***
