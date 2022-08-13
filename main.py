import matplotlib.pyplot as plt
import numpy as np
from numpy import array


plt.plot([1,2,3,4,5,6,7], [1,2,3,5,8,13,20] )
plt.show()

plt.plot([1,2,3,4,5,6,7],  marker= "o" , markersize=10 , linestyle="--",linewidth=3 )

plt.plot([1,2,3,5,8,13,20], marker="o", markersize=10 ,linestyle="--",linewidth=3)

plt.legend(["Normal","Fast"])
plt.show()


plt.figure(figsize=[20,5])
plt.suptitle("my data visulaization assignment")


plt.subplot(1,3,1)
x_plot1 = np.array([1, 2, 3, 4, 5, 6, 7])
y_plot1 = np.array([1, 1, 2, 3, 5, 8, 13])
plt.plot(x_plot1, y_plot1)
plt.show()
plt.title('A Group')

plt.subplot(1,3,2)
plt.title('B Group')
x_plot2 = np.array([0, 1, 2, 3, 4, 5, 6])
y_plot2 = np.array([2, 4, 6, 8, 10, 12, 14])
plt.plot(x_plot2,y_plot2)
plt.show()

plt.subplot(1,3,3)
plt.title('C Group')
x_plot3 = np.array([0, 1, 3, 4])
y_plot3 = np.array([4, 6, 3, 4])
plt.plot(x_plot3,y_plot3)
plt.show()


TV = np.array(["My name" , "Mr robot" ,"Queen of the south"])
Rate = np.array([7.5 , 6 , 9.2])
plt.bar(TV , Rate , color="m" ,width=0.5)
plt.show()


CarAge = np.array([5,7,8,7,2,17,2,9,4,11])
CarSpeed = np.array([99,86,87,88,111,86,103,97,94,78])
color = ["lightcoral" , "darkorange" , "olive" , "violet" , "skyblue" , "teal" , "r" ,"c" , "m" ,"k"]
plt.scatter(CarAge,CarSpeed ,c= color)
plt.show()

x = np.array(["Numpy" , "Pandas" , "Matplotlib" ,"Seaborn" , "Plotly" , "Scikit-learn"])
y = np.array([5,7,6,5,4,1])
color2 = ["lightcoral" , "darkorange" , "olive" , "violet" , "skyblue" , "k"]
plt.pie(y ,labels= x , shadow=True , explode= [0,0.3,0,0,0,0] , colors = color2 ,
        startangle= 180)
plt.show()

x= np.random.randn(250)
print(x)
plt.hist( x , bins= 15)
plt.show()