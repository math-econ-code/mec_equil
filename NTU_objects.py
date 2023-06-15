
import numpy as np
import networkx as nx
import scipy.optimize as opt
from time import time

def list_prefs(x,xletter,yletter,αo_y):
    nby = len(αo_y) - 1
    xsprefs = xletter + str(x) +' : ' 
    for y in range(nby):
        ind = np.where(αo_y== (nby - y ))[0][0]
        if (ind == nby):
            break        
        xsprefs = xsprefs + yletter + str( ind ) + ' > '
    return(xsprefs[:-3])

class NTU_market:
    def __init__(self,α_x_y,γ_x_y,n_x = np.array([]), m_y = np.array([])):
        nbx,nby = α_x_y.shape
        self.α_x_y = np.hstack((α_x_y,np.zeros((nbx,1) )))
        self.γ_x_y = np.vstack((γ_x_y,np.zeros( (1,nby) )))
        if n_x.size == 0:
            n_x = np.ones(nbx)
        if m_y.size ==  0:
            m_y = np.ones(nby) 
        self.n_x,self.m_y = n_x, m_y
        self.largex,self.largey = nby+1, nbx+1
        self.smallx,self.smally = -1, -1
        self.nbx,self.nby = nbx, nby
        self.αo_x_y = np.zeros((nbx,nby+1), dtype = 'int64') 
        self.γo_x_y = np.zeros((nbx+1,nby), dtype = 'int64') 
        self.prefslistα_x_y = np.zeros((nbx,nby+1), dtype = 'int64') 
        self.prefslistγ_x_y = np.zeros((nbx+1,nby), dtype = 'int64') 
        self.traceuo_x_t = np.array([])
        self.traceu_x_t = np.array([])
        for x in range(nbx):
            thelistx = (- self.α_x_y )[x,:].argsort()
            self.αo_x_y[x, thelistx] =  nby  - np.arange(nby+1)
            self.prefslistα_x_y[x,:] = (thelistx) % (self.nby+1) 
        for y in range(nby):
            thelisty = ( - self.γ_x_y)[:,y].argsort()
            self.γo_x_y[thelisty, y] = nbx  - np.arange(nbx+1)
            self.prefslistγ_x_y[:,y] = (thelisty) % (self.nbx+1) 
        self.comp_nbsteps = -1
        self.comp_time = -1
        self.eq_μ_x_y = np.array([])   
        self.eq_u_x = np.array([])
        self.eq_v_y = np.array([])
        

    def print_prefs(self,xs=[],ys=[]):
        if xs == [] and ys==[] :
            xs = range(self.nbx)
            ys = range(self.nby)
        if xs != [] :
            for x in xs :
                print(list_prefs(x,'x','y',self.αo_x_y[x,:]))
            
        if xs != [] and ys != [] :
            print('===')
            
        if ys != [] :
            for y in ys :
                print(list_prefs(y,'y','x',self.γo_x_y[:,y]))
               
           
    def is_GS_stable(self, μ_x_y = None, output=0 ):
        if (min(self.n_x)<1) or (max(self.n_x)>1) or (min(self.m_y)<1) or (max(self.m_y)>1):
            if output > 0 :
                print('n_x or m_y do not only contain ones.')
            return(False)
        if (μ_x_y is None):
            μ_x_y = self.eq_μ_x_y
        μext_x_y0 = np.hstack([μ_x_y, 1-np.sum(μ_x_y,axis = 1).reshape(-1,1) ])
        μext_x0_y = np.vstack([μ_x_y, 1-np.sum(μ_x_y,axis = 0).reshape(1,-1) ])
        if (np.logical_and(μext_x_y0 != 0  , μext_x_y0 != 1 )).any() or (np.logical_and(μext_x0_y != 0  , μext_x0_y != 1 )).any():
            if output > 0 :
                print('The μ is not feasible.')
            return(False)
        uo_x = np.sum(μext_x_y0 * self.αo_x_y, axis = 1)
        vo_y = np.sum(μext_x0_y * self.γo_x_y, axis = 0)
        for x in range(self.nbx):
            for y in range(self.nby):
                if (self.αo_x_y[x,y] > uo_x[x]) and (self.γo_x_y[x,y] > vo_y[y]):
                    if output > 0 :
                        print('The matching is not stable. One blocking pair is: x'+str(x)+',y'+str(y)+'.')
                    return(False)

        for x in range(self.nbx):
            if (self.αo_x_y[x,self.nby] > uo_x[x]):
                if output > 0 :
                    print('The matching is not stable. One blocking pair is: x'+str(x)+',y0.')
                return(False)

        for y in range(self.nby):
            if (self.γo_x_y[self.nbx,y] > vo_y[y]):
                if output > 0 :
                    print('The matching is not stable. One blocking pair is: x0,y'+str(y)+'.')
                return(False)
                
        if output > 0 :
            print('The matching is stable.')
        return (True)

    def μ_from_uo(self, uo_x):
        return np.where( (self.αo_x_y == uo_x.reshape((-1,1))) , 1 , 0 )[:,0:-1]

    def μ_from_vo(self, vo_y):
        return np.where( self.γo_x_y == vo_y , 1 , 0 )[0:-1,:]

    def uo_from_μ(self, μ_x_y = None):
        if (μ_x_y is None):
            μ_x_y = self.eq_μ_x_y
        return np.sum(np.hstack([μ_x_y, (self.n_x-np.sum(μ_x_y,axis = 1)).reshape(-1,1) ]) * self.αo_x_y, axis = 1)

    def vo_from_μ(self, μ_x_y = None):
        if (μ_x_y is None):
            μ_x_y = self.eq_μ_x_y
        return np.sum(np.vstack([μ_x_y, (self.m_y-np.sum(μ_x_y,axis = 0)).reshape(1,-1) ]) * self.γo_x_y, axis = 0)

    def u_from_μ(self, μ_x_y = None):
        if (μ_x_y is None):
            μ_x_y = self.eq_μ_x_y
        return np.sum(np.hstack([μ_x_y, (self.n_x-np.sum(μ_x_y,axis = 1)).reshape(-1,1) ]) * self.α_x_y, axis = 1)

    def v_from_μ(self, μ_x_y = None):
        if (μ_x_y is None):
            μ_x_y = self.eq_μ_x_y
        return np.sum(np.vstack([μ_x_y, (self.m_y-np.sum(μ_x_y,axis = 0)).reshape(1,-1) ]) * self.γ_x_y, axis = 0)

    def solveGaleShapley(self,output=0, trace=False):
        if (output>=2):
            print("Offers made and kept are denoted +1; offers made and rejected are denoted -1.")
        self.comp_nbsteps = 0
        tracemax = self.nbx*self.nby
        if trace:
            self.traceuo_x_t = np.zeros((tracemax,self.nbx))    
        μA_x_y = np.ones((self.nbx+1, self.nby+1), dtype = 'int64') # initially all offers are non rejected
        while True :
            μP_x_y = np.zeros((self.nbx+1, self.nby+1), dtype = 'int64')
            props_x = np.ma.masked_array(self.αo_x_y, μA_x_y[0:-1,:] ==0).argmax(axis = 1) # x's makes an offer to their favorite y
            μP_x_y[range(self.nbx),props_x] = 1 
            μP_x_y[self.nbx,0:self.nby] = 1
            
            μE_x_y = np.zeros((self.nbx+1, self.nby+1), dtype = 'int64') # y's retains their favorite offer among those made:
            rets_y = np.ma.masked_array(self.γo_x_y, μP_x_y[:,0:-1] ==0).argmax(axis = 0)
            μE_x_y[rets_y,range(self.nby)] = 1

            rej_x_y = μP_x_y - μE_x_y # compute rejected offers
            rej_x_y[self.nbx,:] = 0
            rej_x_y[:,self.nby] = 0

            μA_x_y = μA_x_y - rej_x_y  # offers from x that have been rejected are no longer available to x
            if output >= 2:
                print("Round "+str(self.comp_nbsteps)+":\n" , ( (2*μE_x_y-1) * μP_x_y)[0:-1,0:-1])
            if trace and self.comp_nbsteps<tracemax:
                for x in range(self.nbx):
                    self.traceuo_x_t[self.comp_nbsteps,x] = np.sum(self.αo_x_y [x,:] * μP_x_y[x,:])
            self.comp_nbsteps +=1
            if np.max(np.abs(rej_x_y)) == 0: 
                if trace:
                    self.traceuo_x_t = self.traceuo_x_t[0:min(self.comp_nbsteps,tracemax),:]
                break # if all offers have been accepted, then algorithm stops
        self.eq_μ_x_y = μE_x_y[0:-1,0:-1]
        return (0)

    def uo_from_vo(self,vo_y):
        excluded = np.hstack([(self.γo_x_y < vo_y)[0:-1,:],np.zeros((self.nbx,1))])
        return(np.ma.masked_array(self.αo_x_y,  excluded ).max(axis = 1).data)


    def  vo_from_uo(self,uo_x):
        excluded = np.vstack([(self.αo_x_y < uo_x.reshape((-1,1)))[:,0:-1],np.zeros((1,self.nby))])
        return(np.ma.masked_array(self.γo_x_y,  excluded ).max(axis = 0).data)
            
    def solveAdachi(self,output=0,trace=False, startu_x = None, startv_y = None ):
        self.comp_nbsteps = 0
        tracemax = self.nbx*self.nby
        if trace:
            self.traceuo_x_t = np.zeros((tracemax,self.nbx))    
        if startu_x is None:
            uo_x = self.largex * np.ones(self.nbx, dtype = 'int64') # x's utilities are highest
        else:
            uo_x = startu_x
        if startv_y is None:
            vo_y = self.smally * np.ones(self.nby, dtype = 'int64') # y's utilities are lowest
        else:
            vo_y = startv_y
        while True :
            uonew_x = self.uo_from_vo(vo_y) # each x proposes to favorite y among those willing to consider them
            if (uonew_x == uo_x).all() :
                if trace:
                    self.traceuo_x_t = self.traceuo_x_t[0:min(self.comp_nbsteps,tracemax),:]
                break
            uo_x = uonew_x      
            vo_y = self.vo_from_uo(uo_x) # each y proposes to favorite x among those willing to consider them
            if output >= 2:
                μP_x_y = self.μ_from_uo(uo_x)
                μE_x_y = self.μ_from_vo(vo_y)
                print("Round "+str(self.comp_nbsteps)+":\n","μ_P=\n" ,μP_x_y,"\n μ_E=\n", μE_x_y)
            if trace and self.comp_nbsteps<tracemax:
                self.traceuo_x_t[self.comp_nbsteps,:] = uo_x
            self.comp_nbsteps += 1
        self.eq_μ_x_y = self.μ_from_vo(vo_y)
        return(0)

    def Q_z(self,p_z):
        uo_x = p_z[0:self.nbx]
        vo_y = - p_z[self.nbx:(self.nbx+self.nby)]
        uonew_x = self.uo_from_vo(vo_y)
        vonew_y = self.vo_from_uo(uo_x)
        return(np.append(uo_x - uonew_x,vonew_y - vo_y))

    def cux_z(self,p_z):
        uo_x = p_z[0:self.nbx]
        vo_y = - p_z[self.nbx:(self.nbx+self.nby)]
        uonew_x = self.uo_from_vo(vo_y)
        return (np.append(uonew_x , - vo_y))

    def cuy_z(self,p_z):
        uo_x = p_z[0:self.nbx]
        vo_y = - p_z[self.nbx:(self.nbx+self.nby)]
        excluded = np.vstack([(self.αo_x_y < np.repeat([uo_x],self.nby+1,axis = 1).reshape((self.nbx,-1)))[:,0:-1],np.repeat(False,self.nby).reshape((-1,self.nby))])
        vonew_y = self.vo_from_uo(uo_x)
        return( np.append(uo_x, - vonew_y ) )

    def solveCU(self,output=0,trace=False):
        self.comp_nbsteps = 0
        tracemax = self.nbx*self.nby
        if trace:
            self.traceuo_x_t = np.zeros((tracemax,self.nbx))    
        p_z = np.append(self.largex * np.ones(self.nbx), - self.smally * np.ones(self.nby) )
        while True :
            pnew_z = self.cux_z(p_z) 
            pnew_z = self.cuy_z(pnew_z)
            if (pnew_z == p_z).all() :
                if trace:
                    self.traceuo_x_t = self.traceuo_x_t[0:min(self.comp_nbsteps,tracemax),:]
                break
            p_z = pnew_z      
            if trace and self.comp_nbsteps<tracemax:
                self.traceuo_x_t[self.comp_nbsteps,:] = p_z[0:self.nbx]
            self.comp_nbsteps += 1
        self.eq_μ_x_y = self.μ_from_vo(- p_z[self.nbx:(self.nbx+self.nby)])
        return(0)

    def damped_cux_z(self,p_z):
        uo_x = p_z[0:self.nbx]
        vo_y = - p_z[self.nbx:(self.nbx+self.nby)]
        uonew_x = self.uo_from_vo(vo_y)
        return (np.append(uo_x - np.where(uo_x > uonew_x, 1,0)+ np.where(uo_x < uonew_x, 1,0) , - vo_y))

    def solveDampedCU(self,output=0,trace=False):
        self.comp_nbsteps = 0
        tracemax = self.nbx*self.nby
        if trace:
            self.traceuo_x_t = np.zeros((tracemax,self.nbx))    
        p_z = np.append(self.largex * np.ones(self.nbx), - self.smally * np.ones(self.nby) )
        while True :
            pnew_z = self.damped_cux_z(p_z)   # each x proposes to favorite y among those willing to consider them:
            pnew_z = self.cuy_z(pnew_z)
            if (pnew_z == p_z).all() :
                if trace:
                    self.traceuo_x_t = self.traceuo_x_t[0:min(self.comp_nbsteps,tracemax),:]
                break
            p_z = pnew_z      
            # each y proposes to favorite x among those willing to consider them:
            if trace and self.comp_nbsteps<tracemax:
                self.traceuo_x_t[self.comp_nbsteps,:] = p_z[0:self.nbx]
            self.comp_nbsteps += 1
        self.eq_μ_x_y = self.μ_from_vo(- p_z[self.nbx:(self.nbx+self.nby)])
        return(0)


    def solveDeferredAcceptance(self, algorithm = 'Adachi', output=0, trace=False):
        start_time = time()
        if algorithm == 'GS' :
            self.solveGaleShapley(output,trace)
        elif algorithm == 'Adachi' :
            self.solveAdachi(output,trace)
        elif algorithm == 'DARUM' :
            self.solveDARUM(output,trace)   
        elif algorithm == 'CU' :
            self.solveCU(output,trace)
        else:
            raise Exception("Algorithm " + algorithm + " is not implemented.")
        self.comp_time =  time() - start_time
        if output >= 1:
            print("Converged in ",self.comp_nbsteps," steps and ", self.comp_time, " seconds.")
        if output == 1:
            print("mu_x_y=",self.eq_μ_x_y) 
        return (0)

       
    def matched_pairs(self, μ_x_y = None):
        if μ_x_y is None:
            μ_x_y = self.eq_μ_x_y
        nzx,nzy = np.nonzero(μ_x_y)
        return [(nzx[i],nzy[i]) for i in range(len(nzx) )]


    def next_us(self,uo_x,down = True):
        μ_x_y = self.μ_from_uo(uo_x)
        vo_y = self.vo_from_uo(uo_x)
        if down:
            incr= -1
        else:
            incr = 1
        g = nx.DiGraph()
        xs = list(range(self.nbx))
        ys = list(range(self.nbx,self.nby))
        g.add_nodes_from(xs+ys)
        g.add_edges_from([(x,self.nbx+y) for x,y in self.matched_pairs(μ_x_y)])
        #μprime_x_y = np.where( (self.αo_x_y.T == uo_x + incr).T , 1 , 0 )[:,0:-1]
        μprime_x_y = np.where( (self.αo_x_y[:,0:-1].T == uo_x + incr).T &  (self.γo_x_y[0:-1,:]*incr <= vo_y * incr), 1 , 0 ) 
        compatible_pairs = self.matched_pairs(μprime_x_y)
        g.add_edges_from([(self.nbx+j,i) for (i,j) in compatible_pairs])
        cycles = list(nx.simple_cycles(g))
        newus = []
        for c in cycles:
            xs_to_update = [x for x in c if (x <self.nbx)]
            nextu_x = uo_x.copy()
            nextu_x[xs_to_update] += incr
            newus.append(nextu_x)
        return(newus)

    def enumerate_us(self):
        self.solveGaleShapley()
        uo_x = np.sum(self.αo_x_y [:,0:-1] * self.eq_μ_x_y,axis = 1)
        us_list = [uo_x.tolist()]
        i = 0
        while (i < len(us_list)):
            inds_alreadyhere = []
            us_toadd = [arr.tolist() for arr in self.next_us(np.array(us_list[i],dtype='int64'))]
            for u_x in us_toadd:
                if u_x in us_list:
                    inds_alreadyhere.append(us_list.index(u_x))
                    us_toadd.remove(u_x)
            
            inds_toadd = list(range(len(us_list),len(us_list)+len(us_toadd) ))
            us_list = us_list + us_toadd 
            i += 1
        return(np.array([u for u in us_list],dtype = 'int64') )
        
    def DTF(self ,u_x,v_y):
        return np.maximum(u_x.reshape((-1,1)) - self.α_x_y[:,0:-1], v_y - self.γ_x_y[0:-1,:])
            
    def is_aggregate_stable(self, μ_x_y =None ,u_x=None,v_y =None, output=0 ):
        if μ_x_y is None:
            μ_x_y = self.eq_μ_x_y
        if u_x is None:
            u_x = self.eq_u_x
        if v_y is None:
            v_y = self.eq_v_y
        if (np.min(μ_x_y) < 0) or (np.min(self.n_x-np.sum(μ_x_y,axis = 1))<0) or (np.min(self.m_y-np.sum(μ_x_y,axis = 0))<0) :
            if output > 0 :
                print('The μ is not feasible.')
            return(False)
        if (np.min(u_x)<0) or  (np.min(v_y)<0):
            if output > 0 :
                print('u_x < 0 for some x or v_y < 0 for some y.')
            return(False)
        D_x_y = self.DTF(u_x,v_y)
        if np.min(D_x_y)<0 :
            if output > 0 :
                print('There is a blocking pair.')
            return(False)
        if np.sum(μ_x_y * D_x_y)>0 :
            if output > 0 :
                print('Complementary slackness does not hold.')
            return(False)
        if output > 0 :
            print('The matching is stable.')
        return (True)
        
    def aggregateChoice_noHet(self,axis,μbar_x_y):
        if axis == 0 : # if proposing side = x
            n_x = self.n_x
            nbx, nby = self.nbx, self.nby
            prefs_x_y,α_x_y = self.prefslistα_x_y, self.α_x_y
        else:
            n_x = self.m_y
            nbx, nby = self.nby, self.nbx
            prefs_x_y, α_x_y = self.prefslistγ_x_y.T, self.γ_x_y.T
        μ_x_y = np.zeros((nbx,nby+1))
        u_x = np.zeros(nbx)
        for x in range(nbx):
            nxres = n_x[x]
            for yind in prefs_x_y[x,]:
                if yind == nby:
                    μ_x_y[x,yind] = nxres
                    break
                if μbar_x_y[x,yind] > 0:
                    μ_x_y[x,yind] = min(nxres , μbar_x_y[x,yind])
                    nxres -= μ_x_y[x,yind]
                if nxres == 0:
                    break
            u_x[x]=α_x_y[x,yind]
        return(μ_x_y,u_x)

    def aggregateChoice_logit(self,axis,μbar_x_y):
        if axis == 0 : # if proposing side = x
            n_x = self.n_x
            nbx = self.nbx
            nby = self.nby
            α_x_y = self.α_x_y
        else:
            n_x = self.m_y
            nbx = self.nby
            nby = self.nbx
            α_x_y = self.γ_x_y.T    
        μ_x_y = np.zeros((nbx,nby+1))
        u_x = np.zeros(nbx)
        for x in range(nbx):
            thesolμ = opt.brentq(lambda theμ : (theμ+ np.minimum(theμ*np.exp(α_x_y[x,0:-1]),μbar_x_y[x,:]).sum() - n_x[x]) ,0,n_x[x])
            μ_x_y[x,0:-1] = np.minimum(thesolμ*np.exp(α_x_y[x,0:-1]),μbar_x_y[x,:])
            μ_x_y[x,nby] = thesolμ
            u_x[x] = - np.log(thesolμ / n_x[x])
        return(μ_x_y,u_x)

            
    def solveDARUM(self,het1 = 'none',het2 = 'none',output=0,trace=False,tol = 1e-5):
        self.comp_nbsteps = 0
        tracemax = 100*self.nbx*self.nby
        if (output>=2):
            print("Offers made and kept are denoted +1; offers made and rejected are denoted -1.")
        if trace:
            self.traceu_x_t = np.zeros((tracemax,self.nbx))
            self.traceuo_x_t = np.zeros((tracemax,self.nbx))    

        μA_x_y = np.array([[min(self.n_x[x],self.m_y[y]) for y in range(self.nby)] for x in range(self.nbx)]) 
        # initially all offers are non rejected
        while True :
            μP_x_y,self.eq_u_x = self.aggregateChoice(0 , μA_x_y,het1) # the x's pick their preferred offers among those not rejected
            μE_y_x,self.eq_v_y = self.aggregateChoice(1, μP_x_y[:,0:-1].T,het2)
            μE_x_y = μE_y_x.T # the y's pick their preferred offers among those made
            rej_x_y = μP_x_y[:,0:-1] - μE_x_y[0:-1,:] # compute rejected offers
            μA_x_y = μA_x_y - rej_x_y  # offers from x that have been rejected are no longer available to x
            if output >= 2:
                print("Round "+str(self.comp_nbsteps)+":\n" , ( (2*μE_x_y[0:-1,:]-1) * μP_x_y[:,0:-1]))            
            if trace and self.comp_nbsteps < tracemax:
                    self.traceu_x_t[self.comp_nbsteps,:] = self.eq_u_x
                    for x in range(self.nbx):
                        self.traceuo_x_t[self.comp_nbsteps,x] = np.sum(self.αo_x_y [x,:] * μP_x_y[x,:])
            self.comp_nbsteps +=1
            if np.max(np.abs(rej_x_y)) < tol: 
                if trace:
                    self.traceu_x_t = self.traceu_x_t[0:min(self.comp_nbsteps,tracemax),:]
                    self.traceuo_x_t = self.traceuo_x_t[0:min(self.comp_nbsteps,tracemax),:]
                break # if all offers have been accepted (within tolerange), then algorithm stops
        self.eq_μ_x_y = μE_x_y[0:-1,:]
        return (0)


    def aggregateChoice(self,axis,μbar_x_y,heterogeneity = 'none'):
        if heterogeneity == 'none':
            μ_x_y,u_x = self.aggregateChoice_noHet(axis,μbar_x_y)
        elif heterogeneity == 'logit':
            μ_x_y,u_x = self.aggregateChoice_logit(axis,μbar_x_y)
        else:
            raise Exception("Heterogeneity " + heterogeneity + " is not supported.")
        return (μ_x_y,u_x)


    def solveIPFP(self,output=0,trace=False,tol = 1e-5):
        self.comp_nbsteps = 0
        tracemax = 100*self.nbx*self.nby
        if trace:
            self.traceu_x_t = np.zeros((tracemax,self.nbx))
        μ_x0 = np.zeros(self.nbx)
        μ_0y = self.m_y.astype(np.float64)
        while True :
            for x in range(self.nbx):
                μ_x0[x] = opt.brentq (lambda theμ : (theμ+ np.minimum(theμ*np.exp(self.α_x_y[x,0:-1]),μ_0y*np.exp(self.γ_x_y[x,:])).sum() - self.n_x[x]) ,0,1.1*self.n_x[x])
            μP_x_y = np.minimum(μ_x0.reshape((-1,1))*np.exp(self.α_x_y[:,0:-1]),μ_0y*np.exp(self.γ_x_y[0:-1,:]))        
            self.eq_u_x = -np.log(μ_x0 / self.n_x)
            for y in range(self.nby):
                μ_0y[y] = opt.brentq(lambda theμ : (theμ+ np.minimum(μ_x0*np.exp(self.α_x_y[:,y]),theμ*np.exp(self.γ_x_y[0:-1,y])).sum() - self.m_y[y]) ,0,1.1*self.m_y[y])
            μE_x_y = np.minimum(μ_x0.reshape((-1,1))*np.exp(self.α_x_y[:,0:-1]),μ_0y*np.exp(self.γ_x_y[0:-1,:]))        
            self.eq_v_y = -np.log(μ_0y / self.m_y)
            rej_x_y = μP_x_y - μE_x_y # compute rejected offers
            if output >= 2:
                print('μP_x_y=\n',μP_x_y)            
                print('μE_x_y=\n',μE_x_y)            
            if trace and self.comp_nbsteps < tracemax:
                self.traceu_x_t[self.comp_nbsteps,:] = self.eq_u_x
            self.comp_nbsteps +=1
            if np.max(np.abs(rej_x_y)) < tol: 
                if trace:
                    self.traceu_x_t = self.traceu_x_t[0:min(self.comp_nbsteps,tracemax),:]
                break # if all offers have been accepted (within tolerance), then algorithm stops
        self.eq_μ_x_y = μE_x_y
        return (0)


class Aligned_NTU_market(NTU_market):
    def __init__(self,ϕ_x_y):
        NTU_market.__init__(self,α_x_y = ϕ_x_y, γ_x_y = ϕ_x_y)
        self.ϕ_x_y = ϕ_x_y
        self.n,_ =  ϕ_x_y.shape
        ϕo_x_y = np.zeros(self.n* self.n)
        ϕo_x_y[ϕ_x_y.flatten().argsort()] = 1+np.arange(self.n*self.n)
        self.ϕo_x_y=ϕo_x_y.reshape((self.n,self.n))
        
    def solveMaxMaxLex(self):
        theϕo_x_y = self.ϕo_x_y.copy()
        self.eq_μ_x_y = np.zeros((self.n,self.n), dtype = 'int64')
        for k in range(self.n):
            x,y = np.unravel_index(theϕo_x_y.argmax(),(self.n,self.n))
            self.eq_μ_x_y[x,y]=1
            theϕo_x_y[x,:]=0
            theϕo_x_y[:,y]=0
        return 0
        