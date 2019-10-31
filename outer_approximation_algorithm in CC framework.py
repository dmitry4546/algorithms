import collections 
import cvxpy as cvx
import scipy as sp
import scipy.optimize
from scipy.optimize import minimize_scalar
import os
import csv
from time import gmtime, strftime, localtime
import time
import random
from gurobipy import *
import numpy as np
import copy
import pickle
import math
from sklearn import cross_validation
import collections 

def lineno():
    return inspect.currentframe().f_back.f_lineno
class TimeProfiler(object):
    def __enter__(self):
        self._startTime = time.time()
    def __exit__(self):
        return format(time.time() - self._startTime)
class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()       
    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)
class SimulationLC_RCS():
    def __init__(self):
        n=5
        self.n=n
        self.Sigma=range(1,n)
        self.Sigma[n-2]=0
        self.P={}
        self.P[0]=np.ones(n-1)*0.6
        self.P[0]=list(np.random.uniform(0,1,n-1))
        self.P[0][0]=1
        self.P[1]=self.P[0]
        self.P[1][0]=1        
        self.Member=[0.3,0.7]
        self.Member=np.array(self.Member)
        self.n3=100# number of individuals 
        self.n4=100# number of purchases done 
        self.n5=len(self.P[0]) # number of products 

    def SingleSimulation(self,S,Class):       
        for i in range(self.n5):
            s=self.Sigma[i] # s is the product
            if int(S[s])==0:
                continue
            binary=np.random.binomial(1, self.P[Class][s])
            if int(binary)==1:
                purchase=s
                break             
        return purchase

    def DataSimulate(self):
        individuals={}
        for i in range(self.n3):        
            purchases=np.zeros(self.n4)
            OfferSets=np.zeros((self.n5, self.n4))        
            ClassProb=np.random.multinomial(1,self.Member, size=1)
            Class=ClassProb.argmax()
            for j in range(self.n4):    
                OSsize=random.randrange(1, self.n) # the size of the offer set range
                while np.sum(OfferSets[:,j])!=OSsize:
                    itemOS=random.randrange(0,self.n5)
                    OfferSets[itemOS,j]=1    
                OfferSets[0,j]=1 # one item is always available, no purchase 
                purchases[j]=self.SingleSimulation(OfferSets[:,j],Class)     ##+1 to match hotel data set 
            individuals[i]=[OfferSets,purchases]
        return individuals 
    

class SimulationLC_GCS():
    def __init__(self):
        self.Sigma=[1,2,3,4,5,6,0] # first is the most preferred
        self.Member=[1]
        self.Member=np.array(self.Member)
        self.n3=10# number of individuals 
        self.n4=20# number of purchases done 
        self.n5=len(self.Sigma) # number of products 
        self.lambda1={}
        self.lambda2={}
        self.lambda1[0]=np.zeros(self.n5)
        self.lambda2[0]=np.zeros((self.n5,self.n5))
        for i in range(self.n5):
            self.lambda1[0][i]=2/(self.n5*self.n5+self.n5+0.000)            
            for j in range(self.n5):
                if (i!=j):
                    self.lambda2[0][i,j]=2/(self.n5*self.n5+self.n5+0.000)
                    self.lambda2[0][j,i]=2/(self.n5*self.n5+self.n5+0.000)
    def SingleSimulation(self,S,Class):        
        vMult=np.zeros(self.n5)
        for (index,i) in enumerate(S):
            if (i!=0)&(index!=0):
                vMult[index]+=self.lambda1[Class][index]                    
                for (index1,i1) in enumerate(S):
                    if (i1!=0)&(self.Sigma.index(index)<self.Sigma.index(index1)):
                        vMult[index]+=self.lambda2[Class][index, index1]
                    if (i1==0):
                        vMult[index]+=self.lambda2[Class][index, index1]      
            if (index==0):
                vMult[index]+=self.lambda1[Class][index]
            if (i==0):
                vMult[0]+=self.lambda1[Class][index]                
            for (index2,i2) in enumerate(S):
                if (i2==0)&(i==0)&(index<index2):
                    vMult[0]+=self.lambda2[Class][index, index2]                
        purchase=list(np.random.multinomial(1, vMult)).index(1)
        return purchase

    def DataSimulate(self):
        individuals={}
        for i in range(self.n3):        
            purchases=np.zeros(self.n4)
            OfferSets=np.zeros((self.n5, self.n4))        
            ClassProb=np.random.multinomial(1,self.Member, size=1)
            Class=ClassProb.argmax()
            for j in range(self.n4):    
                OSsize=random.randrange(2, 6) # the size of the offer set range
                while np.sum(OfferSets[:,j])!=OSsize:
                    itemOS=random.randrange(0,self.n5)
                    OfferSets[itemOS,j]=1    
                OfferSets[0,j]=1 # one item is always available, no purchase 
                purchases[j]=self.SingleSimulation(OfferSets[:,j],Class)     ##+1 to match hotel data set 
            individuals[i]=[OfferSets,purchases]
        return individuals 

class LC_GCS_RCS_fit():
    def __init__(self, individuals,LossFunction=None, ModelF=None, OutAppr=None, scale=None, PrecisionMain=None, CPtimeLimit=None, MILPtimeLimit=None, TotOptTimeLimit=None, Bburning=None,MIPupdate=None,RelaxBinary=None, PrecisionMILP=None):
        self.PrecisionMILP=PrecisionMILP        
        self.RelaxBinary=RelaxBinary
        self.MIPupdate=MIPupdate        
        self.Bburning=Bburning
        self.TotOptTimeLimit=TotOptTimeLimit
        self.MILPtimeLimit=MILPtimeLimit
        self.CPtimeLimit=CPtimeLimit        
        self.PrecisionMain=PrecisionMain        
        self.OutAppr=OutAppr        
        self.ModelF=ModelF        
        self.LossFunction=LossFunction        
        self.lambda1=None
        self.lambda2=None
        self.tau=None
        self.delta=None
        self.scale=scale
        if (LossFunction=='Square')|(LossFunction=='SquareAggr'):
            self.scale=1
        self.individuals=individuals       
        self.Nproducts=len(individuals[0][0][:,0]) # number of products
        self.Nindividuals=len(self.individuals)
        s=0
        for i in self.individuals.keys():
            s+=len(individuals[i][1])           
        self.Ntransactions=s
        self.TotA,self.TotB,self.TotC,self.TotAA,self.TotBB,self.TotCC,self.UniqTransNumber,self.UniqTransProb,self.TotPurchase=self.ExtractTotFromData(individuals)
        self.A,self.B=self.ObtainDataMatrix()  # Data for RCS model
    def InitializeAlgorithmRCS(self,individuals):   #RCS
        Npurchases=np.zeros(self.Nproducts)
        lambda1=np.zeros(self.Nproducts)
        Si=np.zeros((self.Nproducts,self.Nproducts))
        s=0
        for i in individuals.keys():
            for j in individuals[i][1]:
                Npurchases[int(j)]+=1
                s+=1
        sigma=np.argsort(-Npurchases) 
        sigma=list(sigma)
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):
                if (sigma.index(i)<sigma.index(j)):
                    Si[i,j]=1
        lambda1 = self.Optimization_CP_RCS(Si)
        estimated_ranking=[]            
        for i in range(self.Nproducts):            
            estimated_ranking.append(0)     
        for i in range(self.Nproducts):
            num_rank=self.Nproducts-1
            for j in range (self.Nproducts):
                if i!=j:
                    num_rank=num_rank-int(Si[i,j])
            estimated_ranking[num_rank]=i    
        return lambda1,estimated_ranking,Si

    def InitializeAlgorithm(self,individuals):
        Npurchases=np.zeros(self.Nproducts)
        lambda1=np.zeros(self.Nproducts)
        lambda2=np.zeros((self.Nproducts,self.Nproducts))
        tau=np.zeros((self.Nproducts,self.Nproducts))
        s=0
        for i in individuals.keys():
            for j in individuals[i][1]:
                Npurchases[int(j)]+=1
                s+=1
        sigma=np.argsort(-Npurchases) 
        sigma=list(sigma)       
        Si=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):
                if (sigma.index(i)<sigma.index(j)):
                    Si[i,j]=1                     
        lambda1,lambda2 = self.Optimization_CP(Si)
        estimated_ranking=[]            
        for i in range(self.Nproducts):            
            estimated_ranking.append(0)     
        for i in range(self.Nproducts):
            num_rank=self.Nproducts-1
            for j in range (self.Nproducts):
                if i!=j:
                    num_rank=num_rank-int(Si[i,j])
            estimated_ranking[num_rank]=i    
        tau=lambda2*Si
        return lambda1,lambda2,tau,estimated_ranking,Si
                       
    def ObtainDataMatrix(self):    # exctract Data for RCS model
        A=np.zeros(self.Nproducts)
        B=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nindividuals):
            for (count,j) in enumerate(self.individuals[i][1]):
                A[int(j)]+=1
                for (count1,k) in enumerate(self.individuals[i][0][:,count]):
                    if (count1!=j)&(int(k)==1):
                        B[int(count1),int(j)]+=1
        return A,B
        
    def ExtractTotFromData(self,individuals):
        s=0 

        for i in range(self.Nindividuals):
            for (count,j) in enumerate(individuals[i][1]):
                s+=1

                for (countM,j1) in enumerate(individuals[i][0][:,count]):
                    if (countM!=int(j))&(int(j1)!=0):
                        s+=1
        TotA=np.zeros((s,self.Nproducts))
        TotB=np.zeros((s,self.Nproducts,self.Nproducts))
        TotC=np.zeros((s,self.Nproducts,self.Nproducts))        
        s=0
        dictd={}
        dictd1={}
        listd=[]
        listd1=[]
        DealWithZeroPurchase={} 
        DealWithZeroPurchase1={}
        for i in range(self.Nindividuals):
            for (count,j) in enumerate(individuals[i][1]):
                DealWithZeroPurchase.setdefault(str(individuals[i][0][:,count]),np.zeros(self.Nproducts))
                DealWithZeroPurchase1.setdefault(str(individuals[i][0][:,count]),np.zeros(self.Nproducts))
                DealWithZeroPurchase[str(individuals[i][0][:,count])][int(j)]=1
                dictd[str(int(j))+'_'+str(individuals[i][0][:,count])]=s
                listd.append(str(int(j))+'_'+str(individuals[i][0][:,count])) #purchase plus offer set
                listd1.append(str(individuals[i][0][:,count])) # offer set
                TotA[s,int(j)]+=1 # TotA[i,k]-for observation i, kth item will take value 1, other items will take value 0
                for (count1,k) in enumerate(individuals[i][0][:,count]):
                    if (int(k)==0)&(count1!=int(j)):
                        TotB[s,int(j),count1]+=1 # TotB[i,j,k] -for observation i, j is purchased item, kth item is 1 if it is not offered, and it's 0 if it was offered
                    if (int(k)==1)&(count1!=int(j)):
                        TotC[s,int(j),count1]+=1 # TotC[i,j,k] -for observation i, j is purchased item, kth item is 1 if it is not offered, and it's 0 if it was offered
                s+=1
                for (countM,j1) in enumerate(individuals[i][0][:,count]):
                    if (countM!=int(j))&(int(j1)!=0):
                        DealWithZeroPurchase1[str(individuals[i][0][:,count])][countM]=1
                        dictd1[str(countM)+'_'+str(individuals[i][0][:,count])]=s
                        TotA[s,countM]+=1
                        for (count1,k) in enumerate(individuals[i][0][:,count]):
                            if (int(k)==0)&(count1!=countM):
                                TotB[s,countM,count1]+=1
                            if (int(k)==1)&(count1!=countM):
                                TotC[s,countM,count1]+=1
                        s+=1
        Option=2
        if Option==1:
            TotPurchase=np.sum(TotA, axis=0)
            dictCount=collections.Counter(listd)  
            dictCount1=collections.Counter(listd1) 
            UniqTransProb=np.zeros(len(dictCount))
            UniqTransNumber=np.zeros(len(dictCount))
            TotAA=np.zeros((len(dictCount),self.Nproducts))
            TotBB=np.zeros((len(dictCount),self.Nproducts,self.Nproducts))
            TotCC=np.zeros((len(dictCount),self.Nproducts,self.Nproducts))
            for (count,i) in enumerate(dictCount.keys()):
                UniqTransProb[count]=dictCount[i]/(0.00+dictCount1[i.split('_')[1]])
                UniqTransNumber[count]=dictCount1[i.split('_')[1]]*dictCount1[i.split('_')[1]]
                TotAA[count,:]=TotA[dictd[i],:]
                TotBB[count,:,:]=TotB[dictd[i],:,:]
                TotCC[count,:,:]=TotC[dictd[i],:,:]
        if Option==2:
            TotPurchase=np.sum(TotA, axis=0)
            dictCount=collections.Counter(listd)  
            dictCount1=collections.Counter(listd1)
            s=0            
            for i in dictCount1.keys():
                for j in range(self.Nproducts):
                    if (int(DealWithZeroPurchase[i][int(j)])==0)&(int(DealWithZeroPurchase1[i][int(j)])==1): # condition to check if the item was never purchased under this offer set
                        s+=1
            UniqTransProb=np.zeros(len(dictCount)+s)
            UniqTransNumber=np.zeros(len(dictCount)+s)
            TotAA=np.zeros((len(dictCount)+s,self.Nproducts))
            TotBB=np.zeros((len(dictCount)+s,self.Nproducts,self.Nproducts))
            TotCC=np.zeros((len(dictCount)+s,self.Nproducts,self.Nproducts))
            count=0
            for i in dictCount1.keys():
                for (count1,j) in enumerate(range(self.Nproducts)):
                    if int(DealWithZeroPurchase[i][int(j)])==1:
                        UniqTransProb[count]=dictCount[str(count1)+'_'+i]/(0.00+dictCount1[i])
                        UniqTransNumber[count]=dictCount1[i]*dictCount1[i]
                        TotAA[count,:]=TotA[dictd[str(count1)+'_'+i],:]
                        TotBB[count,:,:]=TotB[dictd[str(count1)+'_'+i],:,:]
                        TotCC[count,:,:]=TotC[dictd[str(count1)+'_'+i],:,:]
                        count+=1
                    if (int(DealWithZeroPurchase[i][int(j)])==0)&(int(DealWithZeroPurchase1[i][int(j)])==1):
                        UniqTransProb[count]=0
                        UniqTransNumber[count]=dictCount1[i]
                        TotAA[count,:]=TotA[dictd1[str(count1)+'_'+i],:]
                        TotBB[count,:,:]=TotB[dictd1[str(count1)+'_'+i],:,:]
                        TotCC[count,:,:]=TotC[dictd1[str(count1)+'_'+i],:,:]         
                        count+=1
        return TotA,TotB,TotC,TotAA,TotBB,TotCC,UniqTransNumber,UniqTransProb,TotPurchase 
    def LogLikeRCS(self,lambda1,Si): # Log-likelihood for RCS model    
        A=self.A
        B=self.B
        C=lambda1[:,None]*Si
        B1=B*Si
        B2=np.sum(B1,axis=1)        
        q=np.dot(np.log(lambda1),A)+np.sum(np.sum(np.log(self.scale*1-C)*B))
        return q/(self.Ntransactions+0.00)
    def RCSoptimizationSingleShot(self):
        A=self.A
        B=self.B
        lambda1 = cvx.Variable(self.Nproducts)
        tau = cvx.Variable(self.Nproducts,self.Nproducts)
        sigma = cvx.Variable(self.Nproducts,self.Nproducts)
        obj = cvx.Maximize(cvx.sum_entries(cvx.mul_elemwise(A,cvx.log(lambda1)))+cvx.sum_entries(cvx.mul_elemwise(B,cvx.log(1-tau))))
        constraints = [lambda1<=1,
                       lambda1 >= 0, tau<=sigma, tau>=0]
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):             
                constraints.append(tau[i,j]<=lambda1[i])
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):             
                constraints.append(tau[i,j]>=lambda1[i]+sigma[i,j]-1)
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):
                if i != j:
                    constraints.append(sigma[i,j] + sigma[j,i] == 1)
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):
                for k in range(self.Nproducts):
                    if (i != j)&(j!=k)&(k!=i):
                        constraints.append(sigma[i,j] + sigma[j,k] + sigma[k,i] <= 2)
        prob = cvx.Problem(obj, constraints) 
        lambdaF=lambda1.value
        tauF=tau.value        
        sigmaF=sigma.value
        estimated_ranking=[]
        SiRound=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):
                if sigmaF[i,j]>0.5:
                    SiRound[i,j]=1
                else:
                    SiRound[i,j]=0     
        for i in range(self.Nproducts):            
            estimated_ranking.append(0)     
        for i in range(self.Nproducts):
            num_rank=self.Nproducts-1
            for j in range (self.Nproducts):
                if i!=j:
                    num_rank=num_rank-int(SiRound[i,j])
            estimated_ranking[num_rank]=i
        print "delta =", sigmaF
        print "delta1 =", sigmaF.sum(axis=1)       
        print "estimated_ranking =",estimated_ranking
    def GCSoptimizationSingleShot(self):
        TotAA=self.TotAA
        TotBB=self.TotBB
        TotCC=self.TotCC
        UniqTransNumber=self.UniqTransNumber
        UniqTransProb=self.UniqTransProb
        m = Model("qp")
        m.setParam('TimeLimit', self.MILPtimeLimit)
        m.setParam('MIPGap', 1e-6)
        lambda1New = m.addVars(self.Nproducts,vtype=GRB.CONTINUOUS, name="lambda1New")
        lambda2New = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.CONTINUOUS, name="lambda2New")
        tauNew = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.CONTINUOUS, name="tauNew")
        deltaNew = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.BINARY, name="deltaNew")
        Obj=LinExpr()
        for k in range(len(self.UniqTransNumber)):
            part=LinExpr()
            for i in range(self.Nproducts):
                part+=TotAA[k,i]*lambda1New[i] 
                for j in range(self.Nproducts):
                    part+=TotBB[k,i,j]*lambda2New[i,j]                
                    part+=TotCC[k,i,j]*tauNew[i,j]
            Obj+=self.UniqTransNumber[k]*(self.UniqTransProb[k]-part)*(self.UniqTransProb[k]-part)
        m.setObjective(Obj, GRB.MINIMIZE) 
        constr = LinExpr()         
        for i in range(self.Nproducts):
            constr+=lambda1New[i]             
            for k in range(0,i):
                constr+=lambda2New[i,k]
        m.addConstr(constr == 1, "c1_")           
        m.addConstrs( (lambda1New[i] >= self.scale*0.001/self.Nproducts for i in range(self.Nproducts)), "c2")
        m.addConstrs( (lambda2New[i,j] >= 0.00000000 for i in range(self.Nproducts)
                                              for j in range(self.Nproducts) 
                                              ), "c3")
        m.addConstrs((tauNew[j,k] - lambda2New[j,k] <= 0 for j in range(self.Nproducts)
                                        for k in range(self.Nproducts)
                                        ), name='c4')  
        m.addConstrs((tauNew[j,k] -         1*deltaNew[j,k] <= 0 for j in range(self.Nproducts)
                                        for k in range(self.Nproducts)
                                        ), name='c5')  
        m.addConstrs((tauNew[j,k] - lambda2New[j,k]-1*deltaNew[j,k] >= -1*1 for j in range(self.Nproducts)
                                        for k in range(self.Nproducts)
                                        ), name='c6')  
        m.addConstrs((tauNew[j,k] >= 0 for j in range(self.Nproducts)
                                        for k in range(self.Nproducts)
                                        ), name='c7') 
        m.addConstrs((lambda2New[j,k]-lambda2New[k,j] == 0 for j in range(self.Nproducts)
                            for k in range(self.Nproducts)
                            if k != j), name='c8') 
        m.addConstrs((deltaNew[i,j] + deltaNew[j,i] == 1 for i in range(self.Nproducts)
                                    for j in range(self.Nproducts)
                                    if i != j), name='c9') 
        m.addConstrs((deltaNew[i,j] + deltaNew[j,k] + deltaNew[k,i] <= 2 for i in range(self.Nproducts)
                                    for j in range(self.Nproducts)
                                    for k in range(self.Nproducts)
                                    if (i != j)&(j!=k)&(k!=i) ), name='c10')
        m.optimize()
        lambda11=m.getAttr('x', lambda1New)
        lambda22=m.getAttr('x', lambda2New)
        delta1=m.getAttr('x', deltaNew)
        tau1=m.getAttr('x', tauNew)
        lambda111=np.zeros(self.Nproducts)
        lambda222=np.zeros((self.Nproducts,self.Nproducts))
        delta2=np.zeros((self.Nproducts,self.Nproducts))
        tau2=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nproducts):
            lambda111[i]=lambda11[i] #lambda_[i].X
            for j in range(self.Nproducts):            
                lambda222[i,j]=lambda22[i,j]
                delta2[i,j]=delta1[i,j]
                tau2[i,j]=tau1[i,j]
        self.lambda1=lambda111
        self.lambda2=lambda222
        self.tau=tau2
        self.delta=delta2
        estimated_ranking=[]
        SiRound=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):
                if self.delta[i,j]>0.5:
                    SiRound[i,j]=1
                else:
                    SiRound[i,j]=0     
        for i in range(self.Nproducts):            
            estimated_ranking.append(0)     
        for i in range(self.Nproducts):
            num_rank=self.Nproducts-1
            for j in range (self.Nproducts):
                if i!=j:
                    num_rank=num_rank-int(SiRound[i,j])
            estimated_ranking[num_rank]=i
        print "delta =", self.delta
        print "estimated_ranking =",estimated_ranking

    def LogLike(self, lambda1,lambda2,tau,Si):  
        TotAA=self.TotAA
        TotBB=self.TotBB
        TotCC=self.TotCC
        Part1=TotAA*lambda1[None,:]
        Part2=TotBB*lambda2[None,:,:]
        Part3=TotCC*tau[None,:,:]
        Total=Part1+np.sum(Part2,axis=2)+np.sum(Part3,axis=2)
        Total1=np.sum(Total, axis=1)
        if self.LossFunction=='Square':
            q=np.dot(self.UniqTransNumber,(Total1-self.UniqTransProb)**2)                
            return q/(self.Ntransactions+0.00)
        else: 
            q=np.dot(self.UniqTransNumber,np.log(Total1))
            return q/(self.Ntransactions+0.00)
    def Extract_CP(self,Si):
        TotAA=self.TotAA
        TotBB=self.TotBB
        TotCC=self.TotCC
        TotCC1=self.TotCC*Si
        C=np.zeros((len(TotBB[:,0,0]),self.Nproducts,self.Nproducts))
        V=np.zeros((len(TotBB[:,0,0]),(self.Nproducts*self.Nproducts+self.Nproducts)/2))
        for i in range(len(TotBB[:,0,0])):
            C[i,:,:]=TotBB[i,:,:]+np.transpose(TotBB[i,:,:])+TotCC1[i,:,:]+np.transpose(TotCC1[i,:,:])
            V[i,:]=self.VarTr(TotAA[i,:],C[i,:,:])            
        return V
    def Optimization_CP(self,Si):
        TotAA=self.TotAA
        TotBB=self.TotBB
        TotCC=self.TotCC
        V=self.Extract_CP(Si)
        x = cvx.Variable(len(V[0,:]))
        if self.LossFunction=='Square':
            obj = cvx.Minimize(self.UniqTransNumber*cvx.square(self.UniqTransProb-V*x)) # Option 1           
        else:
            obj = cvx.Maximize(self.UniqTransNumber*cvx.log(V*x))
        constraints = [cvx.sum_entries(x)==1*self.scale,
                       x[0:self.Nproducts] >= self.scale*0.001/self.Nproducts,
                       x[self.Nproducts:] >= self.scale*0.00]   
        prob = cvx.Problem(obj, constraints) 
        if self.LossFunction=='Square':
            try:
                prob.solve(solver=cvx.GUROBI,reltol = 1e-6,TimeLimit=self.CPtimeLimit, verbose=True)
            except:                                
                print "GUROBI Failure, use cvxopt solver"
                prob.solve(solver=cvx.CVXOPT,reltol = 1e-6, TimeLimit=self.CPtimeLimit, kktsolver='robust', verbose=True)
        else:
            try:
                prob.solve(solver=cvx.CVXOPT, feastol=10e-6, TimeLimit=self.CPtimeLimit,verbose=True)
            except:                  
                prob.solve(solver=cvx.CVXOPT, feastol=10e-6, TimeLimit=self.CPtimeLimit, kktsolver='robust', verbose=True)
        lambda1,lambda2=self.VarTrRev(np.array(x.value)[:,0])
        return lambda1,lambda2
    def Optimization_CP_RCS(self,Si):
        lambda1=np.array(self.Nproducts)
        A=self.A
        B=self.B        
        B1=B*Si
        B2=np.sum(B1,axis=1)
        lambda1=(A+0.000)/(A+B2+0.000)
        lambda1[np.isnan(lambda1)]=0.5
        lambda1[lambda1==0]=0.001/self.Nproducts
        lambda1[lambda1==1]=1-0.001/self.Nproducts        
        return self.scale*lambda1       
    def MILPoptimizeRCS(self, lambda1,Si,tau,m=None):
        A=self.A
        B=self.B
        if m==None:
            if self.RelaxBinary==True: 
                m = Model("lp")
            else:            
                m = Model("mip1")
                m.setParam('MIPGap', self.PrecisionMILP)
            m.setParam('TimeLimit', self.MILPtimeLimit)
            lambda1New = m.addVars(self.Nproducts,vtype=GRB.CONTINUOUS, name="lambda1New")
            tauNew = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.CONTINUOUS, name="tauNew")
            if self.RelaxBinary==True: 
                deltaNew = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.CONTINUOUS, name="deltaNew")
            else:            
                deltaNew = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.BINARY, name="deltaNew")
            MNew= m.addVar(vtype=GRB.CONTINUOUS, name="MNew")
            m.setObjective(MNew, GRB.MAXIMIZE) 
            if self.RelaxBinary==True: 
                for j in range(len(lambda1.keys())):
                    constr = LinExpr()
                    constr +=self.LogLikeRCS(lambda1[j],Si[j])
                    for i in range(self.Nproducts):
                        constr+=(A[i]*(lambda1New[i]-lambda1[j][i] )/lambda1[j][i])/(self.Ntransactions+0.00)
                        for k in range(self.Nproducts):
                            constr+=-(B[i,k]*(tauNew[i,k]-tau[j][i,k])/(self.scale*1-tau[j][i,k]))/(self.Ntransactions+0.00) #-np.log(self.scale)*deltaNew[i,k]/(self.Ntransactions+0.00)
                    m.addConstr(constr >= MNew, "c0_"+str(j))
            else:
                for j in range(len(lambda1.keys())):
                    constr = LinExpr()
                    constr +=self.LogLikeRCS(lambda1[j],Si[j])
                    for i in range(self.Nproducts):
                        constr+=(A[i]*(lambda1New[i]-lambda1[j][i] )/lambda1[j][i])/(self.Ntransactions+0.00)
                        for k in range(self.Nproducts):
                            constr+=-(B[i,k]*(tauNew[i,k]-lambda1[j][i]*Si[j][i,k])/(self.scale*1-lambda1[j][i]*Si[j][i,k]))/(self.Ntransactions+0.00) #-np.log(self.scale)*deltaNew[i,k]/(self.Ntransactions+0.00)
                    m.addConstr(constr >= MNew, "c0_"+str(j))
            j=len(lambda1.keys())-1
            ValFunc=self.LogLikeRCS(lambda1[j],Si[j])
            m.addConstrs( (lambda1New[i] >= self.scale*0.001/self.Nproducts for i in range(self.Nproducts)), "c2")
            m.addConstrs( (lambda1New[i] <= self.scale*(1-0.001/self.Nproducts) for i in range(self.Nproducts)), "c3")
            m.addConstrs((tauNew[j,k] - lambda1New[j] <= 0 for j in range(self.Nproducts)
                                            for k in range(self.Nproducts)
                                            ), name='c4')  
            m.addConstrs((tauNew[j,k] -         self.scale*deltaNew[j,k] <= 0 for j in range(self.Nproducts)
                                            for k in range(self.Nproducts)
                                            ), name='c5')  
            m.addConstrs((tauNew[j,k] - lambda1New[j]-self.scale*deltaNew[j,k] >= -self.scale*1 for j in range(self.Nproducts)
                                            for k in range(self.Nproducts)
                                            ), name='c6')  
            m.addConstrs((tauNew[j,k] >= 0 for j in range(self.Nproducts)
                                            for k in range(self.Nproducts)
                                            ), name='c7') 
            m.addConstrs((deltaNew[i,j] + deltaNew[j,i] == 1 for i in range(self.Nproducts)
                                        for j in range(self.Nproducts)
                                        if i != j), name='c9') 
            m.addConstrs((deltaNew[i,j] + deltaNew[j,k] + deltaNew[k,i] <= 2 for i in range(self.Nproducts)
                                        for j in range(self.Nproducts)
                                        for k in range(self.Nproducts)
                                        if (i != j)&(j!=k)&(k!=i) ), name='c10')
            for i in range(self.Nproducts):
                lambda1New[i].start=self.lambda1PurchaseRankingRCS[i] #lambda_[i].X
                for j in range(self.Nproducts):            
                    deltaNew[i,j].start=self.SiPurchaseRankingRCS[i,j]
            m.optimize()
        else:
            lambda1New=tupledict()
            tauNew=tupledict()
            deltaNew=tupledict()
            for i in range(self.Nproducts):
                lambda1New[i]=m.getVarByName("lambda1New"+"["+str(i)+"]")
                for j in range(self.Nproducts):                    
                    tauNew[i,j]=m.getVarByName("tauNew"+"["+str(i)+","+str(j)+"]")      
                    deltaNew[i,j]=m.getVarByName("deltaNew"+"["+str(i)+","+str(j)+"]")            
            MNew=m.getVarByName("MNew")
            j=len(lambda1.keys())-1
            ValFunc=self.LogLikeRCS(lambda1[j],Si[j])      
            constr = LinExpr()
            constr +=self.LogLikeRCS(lambda1[j],Si[j])
            for i in range(self.Nproducts):
                constr+=(A[i]*(lambda1New[i]-lambda1[j][i] )/lambda1[j][i])/(self.Ntransactions+0.00)
                for k in range(self.Nproducts):
                    constr+=-(B[i,k]*(tauNew[i,k]-lambda1[j][i]*Si[j][i,k])/(self.scale*1-lambda1[j][i]*Si[j][i,k]))/(self.Ntransactions+0.00) #-np.log(self.scale)*deltaNew[i,k]/(self.Ntransactions+0.00)
            m.addConstr(constr >= MNew, "c0_"+str(j))
            m.update()            
            m.optimize()           
        lambda11=m.getAttr('x', lambda1New)
        delta1=m.getAttr('x', deltaNew)
        tau1=m.getAttr('x', tauNew)
        M1=MNew.X
        lambda111=np.zeros(self.Nproducts)
        delta2=np.zeros((self.Nproducts,self.Nproducts))
        tau2=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nproducts):
            lambda111[i]=lambda11[i] #lambda_[i].X
            for j in range(self.Nproducts):            
                delta2[i,j]=delta1[i,j]
                tau2[i,j]=tau1[i,j]
        self.lambda1=lambda111
        self.delta=delta2
        return lambda111,delta2,tau2,M1,ValFunc,m    
    def MILPoptimize(self, lambda1,lambda2,tau,Si,m=None):
        TotAA=self.TotAA
        TotBB=self.TotBB
        TotCC=self.TotCC
        UniqTransNumber=self.UniqTransNumber
        UniqTransProb=self.UniqTransProb
        if m==None:
            m = Model("mip1")
            m.setParam('TimeLimit', self.MILPtimeLimit)
            m.setParam('MIPGap', self.PrecisionMILP)
            lambda1New = m.addVars(self.Nproducts,vtype=GRB.CONTINUOUS, name="lambda1New")
            lambda2New = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.CONTINUOUS, name="lambda2New")
            tauNew = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.CONTINUOUS, name="tauNew")
            deltaNew = m.addVars(self.Nproducts,self.Nproducts,vtype=GRB.BINARY, name="deltaNew")
            MNew= m.addVar(vtype=GRB.CONTINUOUS, name="MNew")
            if self.LossFunction=='Square':
                m.setObjective(MNew, GRB.MINIMIZE) 
                for j in range(len(lambda1.keys())):
                    Part1=TotAA*lambda1[j][None,:]
                    Part2=TotBB*lambda2[j][None,:,:]
                    Part3=TotCC*tau[j][None,:,:]
                    Total=Part1+np.sum(Part2,axis=2)+np.sum(Part3,axis=2)
                    Total1=np.sum(Total, axis=1)
                    Total2=2*self.UniqTransNumber*(Total1-self.UniqTransProb) 
                    TotA1=np.dot(Total2,TotAA) 
                    TotB1=np.sum(Total2[:,None,None]*TotBB,axis=0)
                    TotC1=np.sum(Total2[:,None,None]*TotCC,axis=0)
                    constr = LinExpr()
                    constr +=self.LogLike(lambda1[j],lambda2[j],tau[j],Si[j])
                    for i in range(self.Nproducts):
                        constr+=(lambda1New[i]-lambda1[j][i])*TotA1[i]/(self.Ntransactions+0.00)
                        for k in range(self.Nproducts):
                            constr+=(lambda2New[i,k]-lambda2[j][i,k])*TotB1[i,k]/(self.Ntransactions+0.00)+(tauNew[i,k]-tau[j][i,k])*TotC1[i,k]/(self.Ntransactions+0.00)
                    m.addConstr(constr <= MNew, "c0_"+str(j))
            else:
                m.setObjective(MNew, GRB.MAXIMIZE) 
                for j in range(len(lambda1.keys())):
                    Part1=TotAA*lambda1[j][None,:]
                    Part2=TotBB*lambda2[j][None,:,:]
                    Part3=TotCC*tau[j][None,:,:]
                    Total=Part1+np.sum(Part2,axis=2)+np.sum(Part3,axis=2)
                    Total1=np.sum(Total, axis=1)
                    Total2=self.UniqTransNumber/(Total1) 
                    #Total3=np.transpose(Total2)
                    TotA1=np.dot(Total2,TotAA) 
                    TotB1=np.sum(Total2[:,None,None]*TotBB,axis=0)
                    TotC1=np.sum(Total2[:,None,None]*TotCC,axis=0)
                    constr = LinExpr()
                    constr +=self.LogLike(lambda1[j],lambda2[j],tau[j],Si[j])
                    for i in range(self.Nproducts):
                        constr+=(lambda1New[i]-lambda1[j][i])*TotA1[i]/(self.Ntransactions+0.00)
                        for k in range(self.Nproducts):
                            constr+=(lambda2New[i,k]-lambda2[j][i,k])*TotB1[i,k]/(self.Ntransactions+0.00)+(tauNew[i,k]-tau[j][i,k])*TotC1[i,k]/(self.Ntransactions+0.00)
                    m.addConstr(constr >= MNew, "c0_"+str(j))
            j=len(lambda1.keys())-1
            ValFunc=self.LogLike(lambda1[j],lambda2[j],tau[j],Si[j])
            constr = LinExpr()         
            for i in range(self.Nproducts):
                constr+=lambda1New[i]             
                for k in range(0,i):
                    constr+=lambda2New[i,k]
            if (self.LossFunction=='Square')|(self.LossFunction=='SquareAggr'):
                m.addConstr(constr == 1, "c1_")           
            else:
                m.addConstr(constr == self.scale, "c1_")
            m.addConstrs( (lambda1New[i] >= self.scale*0.001/self.Nproducts for i in range(self.Nproducts)), "c2")
            m.addConstrs( (lambda2New[i,j] >= 0.00000000 for i in range(self.Nproducts)
                                                  for j in range(self.Nproducts) 
                                                  ), "c3")
            m.addConstrs((tauNew[j,k] - lambda2New[j,k] <= 0 for j in range(self.Nproducts)
                                            for k in range(self.Nproducts)
                                            ), name='c4')  
            if (self.LossFunction=='Square')|(self.LossFunction=='SquareAggr'):
                m.addConstrs((tauNew[j,k] -         1*deltaNew[j,k] <= 0 for j in range(self.Nproducts)
                                                for k in range(self.Nproducts)
                                                ), name='c5')  
                m.addConstrs((tauNew[j,k] - lambda2New[j,k]-1*deltaNew[j,k] >= -1*1 for j in range(self.Nproducts)
                                                for k in range(self.Nproducts)
                                                ), name='c6')  
            else:
                m.addConstrs((tauNew[j,k] -         self.scale*deltaNew[j,k] <= 0 for j in range(self.Nproducts)
                                                for k in range(self.Nproducts)
                                                ), name='c5')  
                m.addConstrs((tauNew[j,k] - lambda2New[j,k]-self.scale*deltaNew[j,k] >= -self.scale*1 for j in range(self.Nproducts)
                                                for k in range(self.Nproducts)
                                                ), name='c6')  
                
            m.addConstrs((tauNew[j,k] >= 0 for j in range(self.Nproducts)
                                            for k in range(self.Nproducts)
                                            ), name='c7') 
            m.addConstrs((lambda2New[j,k]-lambda2New[k,j] == 0 for j in range(self.Nproducts)
                                for k in range(self.Nproducts)
                                if k != j), name='c8') 
            m.addConstrs((deltaNew[i,j] + deltaNew[j,i] == 1 for i in range(self.Nproducts)
                                        for j in range(self.Nproducts)
                                        if i != j), name='c9') 
            m.addConstrs((deltaNew[i,j] + deltaNew[j,k] + deltaNew[k,i] <= 2 for i in range(self.Nproducts)
                                        for j in range(self.Nproducts)
                                        for k in range(self.Nproducts)
                                        if (i != j)&(j!=k)&(k!=i) ), name='c10')
            for i in range(self.Nproducts):
                for j in range(self.Nproducts):            
                    if i!=j:                    
                        lambda2New[i,j].start=self.lambda2PurchaseRankingGCS[i,j]
                        deltaNew[i,j].start=self.SiPurchaseRankingGCS[i,j]
            m.optimize()
        else:
            lambda1New=tupledict()
            lambda2New=tupledict()
            tauNew=tupledict()
            deltaNew=tupledict()
            for i in range(self.Nproducts):
                lambda1New[i]=m.getVarByName("lambda1New"+"["+str(i)+"]")
                for j in range(self.Nproducts):                    
                    lambda2New[i,j]=m.getVarByName("lambda2New"+"["+str(i)+","+str(j)+"]")
                    tauNew[i,j]=m.getVarByName("tauNew"+"["+str(i)+","+str(j)+"]")      
                    deltaNew[i,j]=m.getVarByName("deltaNew"+"["+str(i)+","+str(j)+"]")            
            MNew=m.getVarByName("MNew")
            j=len(lambda1.keys())-1
            ValFunc=self.LogLike(lambda1[j],lambda2[j],tau[j],Si[j])
            if self.LossFunction=='Square':
                Part1=TotAA*lambda1[j][None,:]
                Part2=TotBB*lambda2[j][None,:,:]
                Part3=TotCC*tau[j][None,:,:]
                Total=Part1+np.sum(Part2,axis=2)+np.sum(Part3,axis=2)
                Total1=np.sum(Total, axis=1)
                Total2=2*self.UniqTransNumber*(Total1-self.UniqTransProb) 
                #Total3=np.transpose(Total2)
                TotA1=np.dot(Total2,TotAA) 
                TotB1=np.sum(Total2[:,None,None]*TotBB,axis=0)
                TotC1=np.sum(Total2[:,None,None]*TotCC,axis=0)
                constr = LinExpr()
                constr +=self.LogLike(lambda1[j],lambda2[j],tau[j],Si[j])
                for i in range(self.Nproducts):
                    constr+=(lambda1New[i]-lambda1[j][i])*TotA1[i]/(self.Ntransactions+0.00)
                    for k in range(self.Nproducts):
                        constr+=(lambda2New[i,k]-lambda2[j][i,k])*TotB1[i,k]/(self.Ntransactions+0.00)+(tauNew[i,k]-tau[j][i,k])*TotC1[i,k]/(self.Ntransactions+0.00)
                m.addConstr(constr <= MNew, "c0_"+str(j))            
            else:
                Part1=TotAA*lambda1[j][None,:]
                Part2=TotBB*lambda2[j][None,:,:]
                Part3=TotCC*tau[j][None,:,:]
                Total=Part1+np.sum(Part2,axis=2)+np.sum(Part3,axis=2)
                Total1=np.sum(Total, axis=1)
                Total2=self.UniqTransNumber/(Total1) 
                #Total3=np.transpose(Total2)
                TotA1=np.dot(Total2,TotAA) 
                TotB1=np.sum(Total2[:,None,None]*TotBB,axis=0)
                TotC1=np.sum(Total2[:,None,None]*TotCC,axis=0)
                constr = LinExpr()
                constr +=self.LogLike(lambda1[j],lambda2[j],tau[j],Si[j])
                for i in range(self.Nproducts):
                    constr+=(lambda1New[i]-lambda1[j][i])*TotA1[i]/(self.Ntransactions+0.00)
                    for k in range(self.Nproducts):
                        constr+=(lambda2New[i,k]-lambda2[j][i,k])*TotB1[i,k]/(self.Ntransactions+0.00)+(tauNew[i,k]-tau[j][i,k])*TotC1[i,k]/(self.Ntransactions+0.00)
                m.addConstr(constr >= MNew, "c0_"+str(j))
            m.update()            
            m.optimize()           
        lambda11=m.getAttr('x', lambda1New)
        lambda22=m.getAttr('x', lambda2New)
        delta1=m.getAttr('x', deltaNew)
        tau1=m.getAttr('x', tauNew)
        M1=MNew.X
        lambda111=np.zeros(self.Nproducts)
        lambda222=np.zeros((self.Nproducts,self.Nproducts))
        delta2=np.zeros((self.Nproducts,self.Nproducts))
        tau2=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nproducts):
            lambda111[i]=lambda11[i] #lambda_[i].X
            for j in range(self.Nproducts):            
                lambda222[i,j]=lambda22[i,j]
                delta2[i,j]=delta1[i,j]
                tau2[i,j]=tau1[i,j]
        self.lambda1=lambda111
        self.lambda2=lambda222
        self.tau=tau2
        self.delta=delta2
        return lambda111,lambda222,tau2,delta2,M1,ValFunc,m
    def SampleIterationRCS(self):
        lambda1=np.zeros(self.Nproducts)
        lmax=self.scale        
        for i in range(self.Nproducts):
            lambda1[i]=random.uniform(0,lmax)
        return lambda1      
    def SampleIteration(self):
        lambda1=np.zeros(self.Nproducts)
        lambda2=np.zeros((self.Nproducts,self.Nproducts))
        arr=range((self.Nproducts**2+self.Nproducts)/2)
        arr1=random.sample(arr,len(arr))
        l={}
        lmax=self.scale
        if (self.LossFunction=='Square')|(self.LossFunction=='SquareAggr'):
            lmax=1.000
        for i in arr:
            l[i]=random.uniform(0,lmax)
            lmax=lmax-l[i]
        l[i]=lmax
        count=0
        for i in range(self.Nproducts):
            lambda1[i]=l[arr1[count]]
            count+=1
        for i in range(self.Nproducts):
            for j in range(0,i):
                lambda2[i,j]=l[arr1[count]]
                lambda2[j,i]=l[arr1[count]]
                count+=1
        return lambda1,lambda2      
    def Optimization(self):
        lambda1={}   
        lambda2={}
        tau={}
        Si={}
        M={}
        FeasibleValue={}
        ValFunc={}
        if self.ModelF=='RCS':
            lambda1[0],_,Si[0]=self.InitializeAlgorithmRCS(self.individuals)     
            self.lambda1PurchaseRankingRCS=lambda1[0]
            self.SiPurchaseRankingRCS=Si[0]
            
            ValFunc[0]=self.LogLikeRCS(lambda1[0],Si[0])  
            tau[0]=lambda1[0][:,None]*Si[0]
        else:            
            lambda1[0],lambda2[0],tau[0],_,Si[0]=self.InitializeAlgorithm(self.individuals)                
            self.lambda1PurchaseRankingGCS=lambda1[0]
            self.lambda2PurchaseRankingGCS=lambda2[0]
            self.tauPurchaseRankingGCS=tau[0]
            self.SiPurchaseRankingGCS=Si[0]
            ValFunc[0]=self.LogLike(lambda1[0],lambda2[0],tau[0],Si[0])
        M[0]=0
        FeasibleValue[0]=0
        B=self.Bburning
        for i in range(1,B):
            FeasibleValue[i]=0
            M[i]=0
            if self.ModelF=='RCS':
                lambda1[i]=self.SampleIterationRCS()             
                Si[i]=Si[0]
                tau[i]=lambda1[i][:,None]*Si[i]
            else:
                lambda1[i],lambda2[i]=self.SampleIteration()
                Si[i]=Si[0]
                tau[i]=Si[i]*lambda2[i]            
        individuals=self.individuals    
        LL0=0.5
        LL1=1
        iteration=i
        start_time=time.time()
        timeMILP=0
        m=None
        while (np.abs((LL1-LL0)/LL0)>self.PrecisionMain):
            iteration+=1
            p=TimeProfiler() #### time 
            p.__enter__() #### time 
            if self.ModelF=='RCS':
                lambda1[iteration],Si[iteration],tau[iteration],M[iteration],ValFunc[iteration],m=self.MILPoptimizeRCS(lambda1,Si,tau,m=m)
            else:
                lambda1[iteration],lambda2[iteration],tau[iteration],Si[iteration],M[iteration],ValFunc[iteration],m=self.MILPoptimize(lambda1,lambda2,tau,Si,m=m)
            MILP_passes = p.__exit__() #### time 
            #print "sdfsd",MILP_passes
            timeMILP+=float(MILP_passes)
            LL0=ValFunc[iteration]  
            #LL0=self.LogLike(lambda1[iteration],lambda2[iteration],tau[iteration],Si[iteration])
            LL1=M[iteration]
            print "ValFunc[iteration]", "M[iteration]",LL0,LL1
            time_ellapsed = time.time()-start_time
            
            if self.OutAppr==True:
                if self.ModelF=='RCS':
                    lambda1[iteration]=self.Optimization_CP_RCS(Si[iteration])
                    LL0=self.LogLikeRCS(lambda1[iteration],Si[iteration])
                    FeasibleValue[iteration]=LL0
                else:
                    lambda1[iteration],lambda2[iteration]=self.Optimization_CP(Si[iteration])
                    tau[iteration]=lambda2[iteration]*Si[iteration]
                    LL0=self.LogLike(lambda1[iteration],lambda2[iteration],tau[iteration],Si[iteration])
                    FeasibleValue[iteration]=LL0
            else:
                pass
            if self.MIPupdate==True:
                pass
            else:
                m=None
            if time_ellapsed > self.TotOptTimeLimit:
                break
            print "time_ellapsed =", time_ellapsed
        print "Number of iterations", iteration  
        
        if self.OutAppr==True:            
            for i in range(iteration+1):
                print "M, FeasibleValue", M[i],FeasibleValue[i]
        else:
            print "M dict convergence over iterations", M
        estimated_ranking=[]
        SiRound=np.zeros((self.Nproducts,self.Nproducts))
        for i in range(self.Nproducts):
            for j in range(self.Nproducts):
                if Si[iteration][i,j]>0.5:
                    SiRound[i,j]=1
                else:
                    SiRound[i,j]=0     
        for i in range(self.Nproducts):            
            estimated_ranking.append(0)     
        for i in range(self.Nproducts):
            num_rank=self.Nproducts-1
            for j in range (self.Nproducts):
                if i!=j:
                    num_rank=num_rank-int(SiRound[i,j])
            estimated_ranking[num_rank]=i
        if self.OutAppr==False:        
            if self.ModelF=='RCS':
                lambda1[iteration]=self.Optimization_CP_RCS(Si[iteration])
                FeasibleValue=self.LogLikeRCS(lambda1[iteration],Si[iteration])
                BoundValue=M[iteration]
            else:
                lambda1[iteration],lambda2[iteration]=self.Optimization_CP(Si[iteration])
                FeasibleValue=self.LogLike(lambda1[iteration],lambda2[iteration],tau[iteration],Si[iteration])
                BoundValue=M[iteration]
        else:
            if self.ModelF=='RCS':
                lambda1[iteration]=self.Optimization_CP_RCS(Si[iteration])
                FeasibleValue=self.LogLikeRCS(lambda1[iteration],Si[iteration])
                BoundValue=M[iteration]
            else:
                lambda1[iteration],lambda2[iteration]=self.Optimization_CP(Si[iteration])
                FeasibleValue=self.LogLike(lambda1[iteration],lambda2[iteration],tau[iteration],Si[iteration])
                BoundValue=M[iteration]            
        timeCP=time_ellapsed-timeMILP
        if self.ModelF=='RCS':
            print "Model = ", self.ModelF
            print "Loss Function = ", self.LossFunction
            print "MILP time = ", timeMILP
            print "Convex Programming time = ", timeCP         
            print "Optimization time = ", time_ellapsed
            print "Number of iterations = ", iteration
            print "lambda1 = ",lambda1[iteration]          
            print "Si = ",Si[iteration]       
            print "ranking = ",estimated_ranking
            print "FeasibleValue (Lower Bound)", FeasibleValue            
            print "BoundValue (Upper Bound)", BoundValue      
            DiffBounds=(BoundValue-FeasibleValue+0.000)/FeasibleValue
            print "Algorithm precision", DiffBounds
            return lambda1[iteration],Si[iteration],estimated_ranking,FeasibleValue,BoundValue,DiffBounds, timeMILP, timeCP, time_ellapsed, iteration
        else:            
            print "Model = ", self.ModelF
            print "Loss Function = ", self.LossFunction
            print "MILP time = ", timeMILP
            print "Convex Programming time = ", timeCP         
            print "Optimization time = ", time_ellapsed
            print "Number of iterations = ", iteration
            print "lambda1 = ",lambda1[iteration]          
            print "lambda2 = ",lambda2[iteration]          
            print "tau = ",tau[iteration]          
            print "Si = ",Si[iteration]       
            print "ranking = ",estimated_ranking
            if self.LossFunction=='MLE':
                print "FeasibleValue (Lower Bound)", FeasibleValue            
                print "BoundValue (Upper Bound)", BoundValue      
                DiffBounds=(BoundValue-FeasibleValue+0.000)/FeasibleValue
                print "Algorithm precision", DiffBounds
            else:
                print "FeasibleValue (Upper Bound)", FeasibleValue            
                print "BoundValue (Lower Bound)", BoundValue      
                DiffBounds=(FeasibleValue-BoundValue+0.000)/BoundValue
                print "Algorithm precision", DiffBounds
            return lambda1[iteration],lambda2[iteration],Si[iteration],estimated_ranking,FeasibleValue,BoundValue,DiffBounds, timeMILP, timeCP, time_ellapsed, iteration
                
    def VarTr(self,lambda1,lambda2):
        N=len(lambda1)
        x=np.zeros(N+(N*N-N)/2)
        for j in range(N):
            x[j]=lambda1[j]        
        s=j
        for i in range(N):
            for j in range(0,i):
                s+=1
                x[s]=lambda2[i,j]            
        return x
    def VarTrRev(self,x):
        N1=len(x)
        N=int((2*N1+0.25)**0.5-0.5)
        lambda1=np.zeros(N)
        lambda2=np.zeros((N,N))
        for j in range(N):
            lambda1[j]=x[j]        
        s=j
        for i in range(N):
            for j in range(0,i):
                s+=1
                lambda2[j,i]=x[s]
                lambda2[i,j]=x[s]
        return lambda1,lambda2

if 'Simulate Data According to RCS':
    Object=SimulationLC_RCS()
else:
    Object=SimulationLC_GCS()
individuals=Object.DataSimulate()
RelaxBinary=False
MIPupdate=True # True False
LossFunction='Square' # 'MLE' 'Square' 'SquareAggr'
ModelF='GCS' # 'GCS' 'RCS'
OutAppr=True # True False
scale=10 # to smooth MLE loss function 
PrecisionMain=1e-3 # the relative precision for the main iterations
PrecisionMILP=1e-3
CPtimeLimit=1000 # seconds
MILPtimeLimit=60*60 # seconds
TotOptTimeLimit=2*60 # seconds
Bburning=100 # to initialize MILP gradient
Object1=LC_GCS_RCS_fit(individuals,LossFunction=LossFunction, ModelF=ModelF, OutAppr=OutAppr, scale=scale, PrecisionMain=PrecisionMain, CPtimeLimit=CPtimeLimit, MILPtimeLimit=MILPtimeLimit, TotOptTimeLimit=TotOptTimeLimit, Bburning=Bburning, MIPupdate=MIPupdate,RelaxBinary=RelaxBinary, PrecisionMILP=PrecisionMILP)
if ModelF=='RCS':
    lambda1,Si,estimated_ranking,FeasibleValue,BoundValue,DiffBounds, timeMILP, timeCP, time_ellapsed, iteration=Object1.Optimization()    
else:
    lambda1,lambda2,Si,estimated_ranking,FeasibleValue,BoundValue,DiffBounds, timeMILP, timeCP, time_ellapsed, iteration=Object1.Optimization()


