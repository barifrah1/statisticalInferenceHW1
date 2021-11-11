
import numpy as np
from numpy import random
import random as rand
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pylab
REAL_BETAS = np.array([0.05,0.3,0.2])
X_MEAN = 0
X_STD = 1
NUM_TO_DRAW_A = 20
MAX_TRAIN_SIZE = 1000
TRAIN_SIZE_STEP = 10

def calculateLamda(x1,x2,b=REAL_BETAS):
    result = np.exp(np.dot(b,np.array([1,x1,x2])))
    return result

def drawY(numToDraw,mean,std,b=REAL_BETAS):
    yArray = np.array(np.zeros(numToDraw));
    lamdasArray=np.array(np.zeros(numToDraw));
    normals = random.normal(mean,std,numToDraw*2)
    xArray = np.ndarray((numToDraw,2),np.float)
    for i in range(0,numToDraw*2,2):
        index = int(i/2)
        lamdasArray[index] = calculateLamda(normals[i],normals[i+1],b)
        yArray[index] = random.poisson(lam=lamdasArray[index],size=1)
        xArray[index][0]=normals[i]
        xArray[index][1]=normals[i+1]
    return yArray,xArray,lamdasArray
        
def calculatePoissonLoss(y,realY):
    N = len(y)
    logY = np.log(y)
    yRealLogYy= realY*logY
    sumResult = sum(y - yRealLogYy)
    return sumResult/N


if __name__ == '__main__':
    #draw 20 numbers from poisson distribution into y_arrary
    #xArray is 20 couples of numbers randomly distributed from normal(0,1) distribution
    yArray,xArray,lamdasArray = drawY(NUM_TO_DRAW_A,X_MEAN,X_STD)
    #fitting a poisson regressor with log link function
    clf = linear_model.PoissonRegressor(alpha=0)
    clf.fit(xArray,yArray)
    #printing betas
    betas = [clf.intercept_,clf.coef_[0],clf.coef_[1]]
    print(f"beta0:{clf.intercept_}, beta1:{clf.coef_[0]}, beta2:{clf.coef_[1]}")
    #drawing 10000 examples from using betas
    yArrayTest,xArrayTest,lamdasArrayTest = drawY(10000,X_MEAN,X_STD)
    #poisson loss as found in the internet  - sum(y_i-realY_i*log(y_i))/N
    yTestPred = clf.predict(xArrayTest)
    loss = calculatePoissonLoss(yTestPred,yArrayTest)
    print(f"poisson Loss is: {loss}")
    # poisson loss for differnt n: 10,20, … ,1000
    ranges = []
    losses = []
    for n in range(10,MAX_TRAIN_SIZE,TRAIN_SIZE_STEP):
        yArray,xArray,lamdasArray = drawY(n,X_MEAN,X_STD)
        clf = linear_model.PoissonRegressor(alpha=0)
        clf.fit(xArray,yArray)
        betas = [clf.intercept_,clf.coef_[0],clf.coef_[1]]
        yArrayTest,xArrayTest,lamdasArrayTest = drawY(MAX_TRAIN_SIZE,X_MEAN,X_STD)
        yTestPred = clf.predict(xArrayTest)
        loss = calculatePoissonLoss(yTestPred,yArrayTest)
        ranges.append(n)
        losses.append(loss)
    plt.plot(ranges,losses)
    # naming the x axis
    plt.xlabel('numer of examples in the train set')
    # naming the y axis
    plt.ylabel('poisson loss')
    # giving a title to my graph
    plt.title('poisson loss by train set size')
    # calc the trendline
    z = np.polyfit(ranges, losses, 1)
    p = np.poly1d(z)
    pylab.plot(ranges,p(ranges),"r--")    
    # function to show the plot
    plt.show()
    print(f"test losses are: {losses}")
    
    #As we can see, the loss decreases when the train set size increases.
    #This make sense as long as the train size increases the approximated betas converge to the real betas
    #The test set betas are draw from the real betas distribution so when the train size gets bigger the gap between the distribution gets smaller and thats why the loss decreases two. 
    ranges = []
    poissonLosses = []
    lrLosses = []
    for n in range(10,MAX_TRAIN_SIZE,TRAIN_SIZE_STEP):
        yArray,xArray,lamdasArray = drawY(n,X_MEAN,X_STD)
        #poisson regression
        clf = linear_model.PoissonRegressor(alpha=0)
        clf.fit(xArray,yArray)
        betas_poisson = [clf.intercept_,clf.coef_[0],clf.coef_[1]]
        #linear regression
        lr = linear_model.LinearRegression()
        lr.fit(xArray,yArray)
        betas_lr = [lr.intercept_,lr.coef_[0],lr.coef_[1]]
        yArrayTest,xArrayTest,lamdasArrayTest = drawY(MAX_TRAIN_SIZE,X_MEAN,X_STD)
        yTestPredPoisson = clf.predict(xArrayTest)
        yTestPredLr = lr.predict(xArrayTest)
        poissonLoss = mean_squared_error(yArrayTest,yTestPredPoisson)
        lrLoss = mean_squared_error(yArrayTest,yTestPredLr)
        poissonLosses.append(poissonLoss)
        lrLosses.append(lrLoss)
        ranges.append(n)
    plt.plot(ranges,poissonLosses,label="poisson regression")
    plt.plot(ranges,lrLosses, label="lr")
    # naming the x axis
    plt.xlabel('numer of examples in the train set')
    # naming the y axis
    plt.ylabel('MSE')
    # giving a title to my graph
    plt.title('MSE by train set size')
    plt.legend()
    # calc the trendline
    # z = np.polyfit(ranges, losses, 1)
    # p = np.poly1d(z)
    # pylab.plot(ranges,p(ranges),"r--")    
    # function to show the plot
    plt.show()    