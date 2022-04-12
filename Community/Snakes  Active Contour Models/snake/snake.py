import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import threading as th
from scipy import interpolate as inp

class Snake(th.Thread):
    def __init__(self,ax,canvas,lock):
        super().__init__()
        # for plt
        self.ax=ax
        self.canvas=canvas
        self.lock=lock
        self.extc=1
        # for gaussian filter
        self.sigma=1.0
        self.gaussize=3
        # adjustable params
        self.alpha=0.2
        self.beta=0.2
        self.gamma=1
        self.kappa=0.1
        self.wl=0
        self.we=0.5
        self.wt=0
        self.wv=5
        self.wk=0.5
        # filters
        self.m1=np.array([[-1,1]],np.float32)
        self.m2=np.array([[-1],[1]],np.float32)
        self.m3=np.array([[1,-2,1]],np.float32)
        self.m4=np.array([[1],[-2],[1]],np.float32)
        self.m5=np.array([[1,-1],[-1,1]],np.float32)
        self.gx=np.array([[-1/2,0,1/2]])
        self.gy=np.array([[-1/2],[0],[1/2]])
        # others
        self.snakes=[]
        #self.snakes.append(np.array([[3,4,5,1],[3,4,5,2]],np.float32))
        self.sprs=[]
        self.volcs=[]
        self.eps=1e-5
        self.pchngd=1
        self.interp=0.01
        self.img=None
        self.E_con=None
        self.need_draw=0
        self.cnames=["r","b","g","y"]

    def init_img(self,filename):
        self.img=cv2.imread(filename)
        if self.img is None:
            print("Cannot read image ",filename)
            return
        s=self.img.shape
        if len(s)==3:
            self.img=cv2.cvtColor(self.img,cv2.COLOR_RGB2GRAY)
        self.img=cv2.resize(self.img,(600,480))
        #_,self.img=cv2.threshold(self.img,210,255,cv2.THRESH_BINARY)
        self.imggs=cv2.GaussianBlur(self.img.astype(np.float32),(self.gaussize,self.gaussize),self.sigma)
        self.img=self.imggs
        w,h=self.img.shape
        # index
        sx=np.arange(0,h,1)
        sx=np.tile(sx,[w,1])
        sy=np.arange(0,w,1)
        sy=np.tile(sy,[h,1])
        sy=sy.T
        self.sx=sx.astype(np.float32)
        self.sy=sy.astype(np.float32)
        # E_line
        self.E_line=self.img
        # E_edge
        gx,gy=self.grad(self.img)
        self.E_edge=-1*np.sqrt(gx*gx+gy*gy)
        # E_term
        cx=cv2.filter2D(self.imggs,-1,self.m1,borderType=cv2.BORDER_CONSTANT)
        cy=cv2.filter2D(self.imggs,-1,self.m2,borderType=cv2.BORDER_CONSTANT)
        cxx=cv2.filter2D(self.imggs,-1,self.m3,borderType=cv2.BORDER_CONSTANT)
        cyy=cv2.filter2D(self.imggs,-1,self.m4,borderType=cv2.BORDER_CONSTANT)
        cxy=cv2.filter2D(self.imggs,-1,self.m5,borderType=cv2.BORDER_CONSTANT)
        self.E_term=(cyy*cx*cx-2*cxy*cx*cy+cxx*cy*cy)/((self.eps+cx*cx+cy*cy)**1.5)
        # E_img
        self.E_img=self.wl*self.E_line+self.we*self.E_edge+self.wt*self.E_term
        self.E_con=np.zeros(self.E_img.shape)
        # draw pic
        self.clr()
        self.draw_pic()

    def init_params(self):
        # init alpha and beta
        self.b1=self.beta
        self.b2=-(self.alpha+4*self.beta)
        self.b3=(2*self.alpha+6*self.beta)
        self.b4=self.b2
        self.b5=self.b1
        self.Ainv=[]
        for snake in self.snakes:
            n,m=snake.shape
            e=np.eye(m)
            a=self.b1*np.roll(e,-2,axis=1)
            a=a+self.b2*np.roll(e,-1,axis=1)
            a=a+self.b3*e
            a=a+self.b4*np.roll(e,1,axis=1)
            a=a+self.b5*np.roll(e,2,axis=1)
            ainv=np.linalg.inv(a+self.gamma*e)
            self.Ainv.append(ainv)
        self.pchngd=0

    def init_cons(self):
        # volcano
        n,m=self.img.shape
        self.E_con=np.zeros((n,m))
        for xy in self.volcs:
            x,y=xy
            self.E_con+=(self.wv*10000)/((self.sx-x)**2+(self.sy-y)**2+self.eps)
        self.E_img=self.wl*self.E_line+self.we*self.E_edge+self.wt*self.E_term
        self.E_ext=self.E_img+self.E_con
        self.fx,self.fy=self.grad(self.E_ext)

    def grad(self,x):
        gx=cv2.filter2D(x,-1,self.gx,borderType=cv2.BORDER_REPLICATE)
        gx[:,[0,-1]]*=2
        gy=cv2.filter2D(x,-1,self.gy,borderType=cv2.BORDER_REPLICATE)
        gy[[0,-1]]*=2
        return gx,gy

    def add_s(self,pts):
        if(len(pts)<3):
            return
        x,y=pts[0]
        # for circle
        pts.append([x,y])
        n=len(pts)
        pts=np.array(pts)
        tck,u=inp.splprep([pts[:,0],pts[:,1]],s=1)
        unew=np.arange(0,1,self.interp)
        npts=inp.splev(unew,tck)
        npts=np.array(npts)
        self.snakes.append(npts)
        self.pchngd=1
        self.clr()
        self.draw_pic()

    def add_spr(self,s,d):
        aix=0
        aiy=0
        axs=-1
        for i,snake in enumerate(self.snakes):
            x,y=snake
            sx,sy=s
            dx,dy=d
            sxy=(x-sx)**2+(y-sy)**2
            ix=np.argmin(sxy)
            xy=np.min(sxy)
            if axs==-1 or axs>xy:
                axs=xy
                aix=i
                aiy=ix
        if axs!=-1:
            if axs>200:
                return
            self.sprs.append((aix,aiy,dx,dy))

    def draw_pic(self):
        if self.img is not None:
            self.ax.imshow(self.img,cmap="gray")
        for i,s in enumerate(self.snakes):
            x=s[0]
            y=s[1]
            x=np.append(x,s[0,0])
            y=np.append(y,s[1,0])
            self.ax.plot(x,y,self.cnames[i])
        for xy in self.volcs:
            x,y=xy
            self.ax.scatter(x,y,c = 'b',marker = 'o')

        for sp in self.sprs:
            ix,iy,dx,dy=sp
            sx=self.snakes[ix][0,iy]
            sy=self.snakes[ix][1,iy]
            self.ax.plot([sx,dx],[sy,dy],"go--")
        
        self.canvas.draw()

    def clr(self):
        self.ax.cla()
        self.ax.set_xlim(0,600)
        self.ax.set_ylim(480,0)

    def clr_snake(self):
        self.snakes=[]
        self.clr()
        self.draw_pic()

    def get_params(self,t):
        if t=="alpha":
            return self.alpha
        elif t=="beta":
            return self.beta
        elif t=="gamma":
            return self.gamma
        elif t=="kappa":
            return self.kappa
        elif t=="wl":
            return self.wl
        elif t=="we":
            return self.we
        elif t=="wt":
            return self.wt
        elif t=="wv":
            return self.wv
        elif t=="wk":
            return self.wk
        return None

    def set_params(self,t,x):
        if type(x) is not float:
            x=float(x)
        if t=="alpha":
            self.alpha=x
        elif t=="beta":
            self.beta=x
        elif t=="gamma":
            self.gamma=x
        elif t=="kappa":
            self.kappa=x
        elif t=="wl":
            self.wl=x
        elif t=="we":
            self.we=x
        elif t=="wt":
            self.wt=x
        elif t=="wv":
            self.wv=x
        elif t=="wk":
            self.wk=x
        else:
            return -1
        self.pchngd=1

    def run(self):
        cnt=0
        while self.extc:
            self.lock.acquire()
            self.lock.release()
            self.init_cons()
            if self.pchngd:
                self.init_params()
            n,m=self.fx.shape
            ux=np.arange(0,m,1)
            uy=np.arange(0,n,1)
            fx=inp.RectBivariateSpline(uy,ux,self.fx)
            fy=inp.RectBivariateSpline(uy,ux,self.fy)
            for i,snake in enumerate(self.snakes):
                x=snake[0]
                y=snake[1]
                zx=fx(y,x,grid=False)
                zy=fy(y,x,grid=False)
                for sp in self.sprs:
                    ix,iy,dx,dy=sp
                    if ix==i:
                        zx[iy]+=self.wk*(x[iy]-dx)
                        zy[iy]+=self.wk*(y[iy]-dy)
                nx=self.gamma*x-self.kappa*zx
                ny=self.gamma*y-self.kappa*zy
                snake[0]=np.dot(self.Ainv[i],nx)
                snake[1]=np.dot(self.Ainv[i],ny)
            time.sleep(0.001)
            cnt+=1
            if cnt%2==0:
                self.need_draw=1
                cnt=0
            
