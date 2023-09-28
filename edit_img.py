import numpy as np 
import cv2
import math
from ipywidgets import interact, IntSlider , Select
import matplotlib.pyplot as plt
%matplotlib inline
 
x1 = IntSlider(value=1, min=0, max=20, step=1, description='x:',)
y1 = IntSlider(value=1, min=0, max=20, step=1, description='y:',)
thetaX = IntSlider(value=1, min=-60, max=60, step=1, description='θx:',)
thetaY = IntSlider(value=1, min=-60, max=60, step=1, description='θy:',)
files = ['wiki.png','enjoy.png','ancient_view_QR.jpg']
selected = Select(description='画像', options=files, rows=5,)
 
def show_plot(col1, col2 , col3, col4, col5):
    # -- 画像の読み取り --
    img = cv2.imread('img/' + col3)
    (h, w, c) = img.shape  # 画像のサイズ取得
    # -- 画像のせん断変形 -- 
    theta1 = np.deg2rad(col4)
    theta2 = np.deg2rad(col5)
    mat = np.float32([[1, np.tan(theta1), 0], [np.tan(theta2), 1, 0]])
    scale = 4
    img = cv2.warpAffine(img, mat, (w*scale, h*scale))
    # ---- 画像サイズ再取得 ----
    (h, w, c) = img.shape
    # -- 画像の歪み --
    flex_x = np.zeros((h,w),np.float32)
    flex_y = np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w):
            flex_x[y,x] = x + math.sin(x/30) * col1
            flex_y[y,x] = y + math.cos(y/30) * col2
    dst = cv2.remap(img,flex_x,flex_y,cv2.INTER_LINEAR)
    dst = cv2.cvtColor(dst , cv2.COLOR_BGR2RGB)
    # 画像を中心に移動する
    dx, dy = 350, 350
    # ---- 平行移動の変換行列 ----
    move_matrix = np.float32([[1,0,dx],[0,1,dy]])
    # ---- 平行移動適用 ----
    dst = cv2.warpAffine(dst, move_matrix, (w,h))
    plt.figure(figsize=(8,8))
    plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.imshow(dst)
    plt.show()
 
interact(show_plot, col1=x1, col2=y1 , col3 = selected, col4=thetaX, col5=thetaY)