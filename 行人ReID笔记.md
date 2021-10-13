# 行人ReID笔记

Person ReID: recognition(ID) <-- image retrieval (图像检索) (CBIR--Content-base retrieval)

data: Market-1501

Image-path PersonID (X, y)

Objective: learn a feature space --> Sim(X1, X2) is large if the same person ID

Implementation: Input -> a photo, Output -> feature vector f(X) --> 1024 dim vector

sim{f(X1), f(X2)} = f1 * f2 (内积) / |f1| |f2|

Given a query photo q, and a database {I1, I2, ...}, Top-1 error

f: visual encoder --> CNN

