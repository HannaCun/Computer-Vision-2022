import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from itertools import permutations,combinations
import time
# 将图片转为灰度图
img = cv.imread('2.png', 0)
cv.imshow("img", img)
# cv.waitKey()
# plt.figure()
# plt.hist(img.ravel(),256)
# plt.title('original_hist')



def OTSU(img_gray, GrayScale):
    assert img_gray.ndim == 2, "must input a gary_img"  # shape有几个数字, ndim就是多少
    img_gray = np.array(img_gray).ravel().astype(np.uint8)
    u1 = 0.0  # 背景像素的平均灰度值
    u2 = 0.0  # 前景像素的平均灰度值
    u3 = 0.0
    th1 = 0.0
    th2 = 0.0

    # 总的像素数目
    PixSum = img_gray.size
    # 各个灰度值的像素数目
    PixCount = np.zeros(GrayScale)
    # 各灰度值所占总像素数的比例
    PixRate = np.zeros(GrayScale)
    # 统计各个灰度值的像素个数
    for i in range(PixSum):
        # 默认灰度图像的像素值范围为GrayScale
        Pixvalue = img_gray[i]
        PixCount[Pixvalue] = PixCount[Pixvalue] + 1

    # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
    for j in range(GrayScale):
        PixRate[j] = PixCount[j] * 1.0 / PixSum
    Max_var = 0
    nums = [i for i in range(GrayScale)]
    #print(nums)
    res = comb(nums)
    print("len(res)",len(res))
    lst = []
    lstth1=[]
    lstth2=[]
    for i in range(len(res)):
        #print(i)
        u1_tem = 0.0
        u2_tem = 0.0
        u3_tem = 0.0
        tmp1 = res[i][0]
        tmp2 = res[i][1]
        #print(tmp1,tmp2)
        w1 = np.sum(PixRate[:tmp1])
        w2 = np.sum(PixRate[tmp1:tmp2])
        w3 = 1 - w1 - w2
        #print(type(w3),w3)
        if w1 == 0 or w2 == 0 or w3 == 0 :
            pass
        else:
            # 类别1的平均像素
            for m1 in range(tmp1):
                u1_tem = u1_tem + PixRate[m1] * m1
            u1 = u1_tem * 1.0/w1
            #print(type(u1), u1)
            for m2 in range(tmp1,tmp2):
                u2_tem = u2_tem + PixRate[m2] * m2
            u2 = u2_tem *1.0 /w2

            for m3 in range(tmp2,GrayScale):
                u3_tem = u3_tem + PixRate[m3] * m3
            u3 = u3_tem * 1.0 /w3
            #print(type(u3), u3)
            ut = w1*u1 + w2*u2 + w3*u3
            #print(type(ut),ut)
            tem_var = w1*(u1-ut)*(u1-ut)+w2*(u2-ut)*(u2-ut)+w3*(u3-ut)*(u3-ut)
            lst.append(tem_var)
            lstth1.append(tmp1)
            lstth2.append(tmp2)
            if Max_var<tem_var:
                Max_var = tem_var
                th1 = tmp1
                th2 = tmp2
    # 确定最大类间方差对应的阈值
    return lstth1,lstth2,lst,th1,th2

def comb(nums):
    res = []
    for i in combinations(nums,2):
        res.append(list(i))
    return res
# nums = [i for i in range(4)]
# print(comb(nums))
t=time.time()
lstth1,lstth2,lst,th1,th2 = OTSU(img,256)
t1 = time.time() - t
print(t1)
print(th1,th2)
im = np.array(img)
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if im[i][j] < th1:
            im[i][j] = 0
        elif im[i][j]>=th1 and im[i][j]<th2:
            im[i][j] = 0.5 *(th1 + th2)
        else:
            im[i][j] = 255
print(len(lstth1),len(lstth2),len(lst))
fig = plt.figure()
ax3 = plt.axes(projection='3d')
xx = np.array(lstth1)
yy = np.array(lstth2)
Z = np.array(lst)

ax3.scatter3D(xx,yy,Z,c='green',s=5)
ax3.set_xlabel('threshold1')
ax3.set_ylabel('threshold2')
ax3.set_zlabel('class_var')

plt.figure()
plt.imshow(Image.fromarray(im) , cmap='gray')
plt.title('OTSU_threeThresholds')
plt.figure()
plt.hist(im.ravel(),256)
plt.title('hist')

plt.show()