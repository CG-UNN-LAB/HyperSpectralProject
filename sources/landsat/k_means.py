import random
import numpy as np
import matplotlib.pyplot as plt


N=800
k=6
      
        
n= float(N)/k # среднее, все точки делить на кол-во кластеров, среднее кол-во точек для одного кластера
X=[]
for i in range(k):
    c = (random.uniform(-1,1), random.uniform(-1,1))# uniform принимает от скольки до скольки(интервал) и размер. Это кортеж
    s=random.uniform(0.05,0.5) # значение
    x=[]
    while len(x)<n:
        a=np.random.normal(c[0], s)# normal принимает центр распределения, ширина отклонения
        b=np.random.normal(c[1], s)# normal принимает центр распределения, ширина отклонения
#        a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
        #Продолжаем рисовать точки из распределения в диапазоне [-1,1]
        if abs(a)<1 and abs(b)<1:# возвращает модуль числа
            x.append([a,b])
    X.extend(x)# удлиняет список, добавлят элементы, конкретно a и b, а не [a,b]
    
z=np.zeros( ( 40,40) ) 

X = np.array(X)[:N]

X=np.array(X).reshape( ( 40,40 ) ) 
print(np.shape(X))
for i in range(40):
    for j in range(40):
        k=X[i,j]
        z[i,j]=k




fig=plt.figure(figsize=(8, 8))


fig.add_subplot(2, 1, 1)
plt.imshow(z1)

fig=plt.figure(figsize=(8, 8))


fig.add_subplot(2, 1, 2)
plt.imshow(z) 

plt.show()






    
    




    



