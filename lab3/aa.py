import cv2
import numpy as np
import matplotlib.pyplot as plt

FACE_PATH = "orl_faces"
PERSON_NUM = 40
PERSON_FACE_NUM = 10
K = 30  # Number of principle components
show_K = 30

raw_img = []#原始图像每个人的一张
data_set = []#gray & reshape(height*width)
data_set_label = []
def read_data():
    for i in range(1, PERSON_NUM + 1):
        person_path = FACE_PATH + '/s' + str(i)
        for j in range(1, PERSON_FACE_NUM + 1):
            img = cv2.imread(person_path + '/' + str(j) + '.pgm')
            if j == 1:
                raw_img.append(img)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img_gray.shape
            img_col = img_gray.reshape(height * width)
            data_set.append(img_col)
            data_set_label.append(i)
    return height, width

# 导入数据集
height, width = read_data()
X = np.array(data_set)
Y = np.array(data_set_label)
n_sample, n_feature = X.shape

## 实现pca 算法步骤：
# X 400*10304# 第一步:每个像素求均值  #压缩行，对各列求均值，返回 1* n 矩阵
average_face = np.mean(X, axis=0)
equalization_X = X - average_face# 减去对应的均值 #行是sample
equalization_X = equalization_X.T #10304*400
#利用svd分解 不必求解XX^T而是直接通过分解 得到U 取U的前P个特征向量即可得到特征向量矩阵
U,s,V = np.linalg.svd(equalization_X)
U=U.T#######注意U U的每一列才是特征向量 每一行得到的效果偏向原图
#选择一个K .0 **2 K/5
sum=0.0
temp=0.0
for i in s:
   sum+=i**2
for i in range(len(s)):
    temp+=s[i]**2
    if temp>=0.95*sum:
        K=i+1
        break
print("K = ",K)
U = U[:K]
print(U.shape) #10*10304

#打印 特征向量矩阵 图片
figure = plt.figure()
figure.suptitle('eigenvectors matrix K = '+str(K))
for i in range(0, show_K):
    plt.subplot(int(show_K/5), 5, i+1)
    plt.imshow(U[i].reshape(height, width), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
plt.show()

Y = U.dot(equalization_X) #正变换 10*400
X_ni = U.T.dot(Y).T+average_face #400*10304 +10304才对哈 第二维度要相等，而且reshape的时候也是这样
#打印逆变换后的图片
X_after_img = X_ni.reshape((400, height, width))  # 特征脸
figure = plt.figure()
figure.suptitle('Contravariant image K = '+str(K))
for i in range(0, show_K):
    #plt.title(title_graph)
    plt.subplot(int(show_K/5), 5, i+1)
    plt.imshow(X_after_img[i], cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
plt.show()
