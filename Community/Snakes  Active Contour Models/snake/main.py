from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as fctk
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
import time
import numpy as np
from snake import Snake
from threading import Lock

class MainWindow():
    def __init__(self,init_window=Tk()):
        self.rt_window = init_window
        self.set_init_window()
        self.lock=Lock()
        self.snakes=Snake(self.ax,self.canvas,self.lock)
        self.init_scales()
        self.pts=[]
        self.spr=[]
        self.blck=0
        self.drawp=None

    def __del__(self):
        self.snakes.extc=0

    def start(self):
        self.rt_window.mainloop()

    # init window
    def set_init_window(self):
        self.rt_window.title("Snake V_0.5")
        self.rt_window.geometry('760x600+100+100')
        
        # setting frame
        self.frm_t=Frame(self.rt_window, width=760, height=480)
        self.frm_b=Frame(self.rt_window, relief="raised", bd=3, width=760, height=120)
        self.frm_l=Frame(self.frm_t, width=160, height=480)
        self.frm_r=Frame(self.frm_t, width=600, height=480)
        self.frm_t.grid_propagate(False)
        self.frm_b.grid_propagate(False)
        self.frm_l.grid_propagate(False)
        self.frm_r.grid_propagate(False)
        self.frm_t.grid(row=0,column=0)
        self.frm_b.grid(row=1,column=0)
        self.frm_l.grid(row=0,column=0)
        self.frm_r.grid(row=0,column=1)
        
        self.frm_file=Frame(self.frm_l, relief="raised", bd=3 ,width=160, height=100)
        self.frm_tool=Frame(self.frm_l, relief="raised", bd=3 ,width=160, height=380)
        self.frm_file.grid_propagate(False)
        self.frm_tool.grid_propagate(False)
        self.frm_file.grid(row=0,column=0)
        self.frm_tool.grid(row=1,column=0)

        # botton
        self.button1=Button(self.frm_tool,text="snake",width=5,command=self.draw_snake,cursor="hand2",state="disabled")
        self.button1.grid(row=0,column=0,padx=25,pady=10)
        self.button2=Button(self.frm_tool,text="pause",width=5,command=self.pauser,cursor="hand2",state="disabled")
        self.button2.grid(row=0,column=1)
        self.button3=Button(self.frm_tool,text="volca",width=5,command=self.add_volc,cursor="hand2",state="disabled")
        self.button3.grid(row=1,column=0)
        self.button4=Button(self.frm_tool,text="spring",width=5,command=self.add_spr,cursor="hand2",state="disabled")
        self.button4.grid(row=1,column=1)
        self.button_open=Button(self.frm_file,text="Open",width=5,command=self.openfile,cursor="hand2")
        self.button_open.grid(row=0,column=0,padx=50,pady=10)
        self.button_clear=Button(self.frm_file,text="Clear",width=5,command=self.clr,cursor="hand2")
        self.button_clear.grid(row=1,column=0,padx=50)

        # Canvas
        self.img=''
        self.pltfig=Figure(figsize=(5,4),dpi=120)
        self.canvas=fctk(self.pltfig,master=self.frm_r)
        self.canvas.get_tk_widget().grid(row=0,column=0)
        self.ax=self.pltfig.add_axes([0,0,1,1])

    def init_scales(self):
        # scale
        self.scale1=Scale(self.frm_tool,orient = HORIZONTAL,from_=0.0,to=1.0,resolution=-1,label="alpha",
                          tickinterval=0,command=lambda x:self.snakes.set_params("alpha",x))
        self.scale1.set(self.snakes.get_params("alpha"))
        self.scale1.grid(row=2,column=0,columnspan=5,padx=10,pady=(0,0))
        
        self.scale2=Scale(self.frm_tool,orient = HORIZONTAL,from_=0.0,to=1.0,resolution=-1,label="beta",
                          tickinterval=0,command=lambda x:self.snakes.set_params("beta",x))
        self.scale2.set(self.snakes.get_params("beta"))
        self.scale2.grid(row=3,column=0,columnspan=5,padx=10)
        
        self.scale3=Scale(self.frm_tool,orient = HORIZONTAL,from_=0.0,to=3.0,resolution=-1,label="gamma",
                          tickinterval=1.5,command=lambda x:self.snakes.set_params("gamma",x))
        self.scale3.set(self.snakes.get_params("gamma"))
        self.scale3.grid(row=4,column=0,columnspan=5,padx=10)
        
        self.scale4=Scale(self.frm_tool,orient = HORIZONTAL,from_=0.0,to=1.0,resolution=-1,label="kappa",
                          tickinterval=0,command=lambda x:self.snakes.set_params("kappa",x))
        self.scale4.set(self.snakes.get_params("kappa"))
        self.scale4.grid(row=5,column=0,columnspan=5,padx=10,)
        
        self.scale5=Scale(self.frm_b,orient = HORIZONTAL,from_=-2.0,to=2.0,resolution=-1,label="wline",
                          length=125,tickinterval=2,command=lambda x:self.snakes.set_params("wl",x))
        self.scale5.set(self.snakes.get_params("wl"))
        self.scale5.grid(row=0,column=0,padx=10,pady=15)
        self.scale6=Scale(self.frm_b,orient = HORIZONTAL,from_=-2.0,to=2.0,resolution=-1,label="wedge",
                          length=125,tickinterval=2,command=lambda x:self.snakes.set_params("we",x))
        self.scale6.set(self.snakes.get_params("we"))
        self.scale6.grid(row=0,column=1,padx=10)
        self.scale7=Scale(self.frm_b,orient = HORIZONTAL,from_=-2.0,to=2.0,resolution=-1,label="wterm",
                          length=125,tickinterval=2,command=lambda x:self.snakes.set_params("wt",x))
        self.scale7.set(self.snakes.get_params("wt"))
        self.scale7.grid(row=0,column=2,padx=10)
        self.scale8=Scale(self.frm_b,orient = HORIZONTAL,from_=-10,to=10,resolution=-1,label="volcano",
                          length=125,tickinterval=10,command=lambda x:self.snakes.set_params("wv",x))
        self.scale8.set(self.snakes.get_params("wv"))
        self.scale8.grid(row=0,column=3,padx=10)
        self.scale9=Scale(self.frm_b,orient = HORIZONTAL,from_=0,to=1.0,resolution=-1,label="spring",
                          length=125,tickinterval=1,command=lambda x:self.snakes.set_params("wk",x))
        self.scale9.set(self.snakes.get_params("wk"))
        self.scale9.grid(row=0,column=4,padx=10)

    def openfile(self):
        filename=askopenfilename()
        if len(filename)==0:
            return
        print(filename)
        self.snakes.init_img(filename)
        #self.snakes.calc()
        if not self.snakes.is_alive():
            self.snakes.start()
        self.updter()
        self.button1["state"]="normal"
        self.button2["state"]="normal"
        self.button3["state"]="normal"
        self.button4["state"]="normal"

    def clr(self):
        self.snakes.clr_snake()

    def updter(self):
        self.rt_window.after(20,self.updter)
        #print(self.snakes.need_draw)
        if self.snakes.need_draw:
            self.snakes.clr()
            if self.drawp is not None:
                self.ax.scatter(self.drawp[0],self.drawp[1],c = 'g',marker = 'o')
            self.snakes.draw_pic()
            self.snakes.need_draw=0

    def pauser(self):
        if self.button2["text"]=="pause":
            self.button2["text"]="resume"
            self.lock.acquire()
        else:
            self.button2["text"]="pause"
            self.snakes.extc=1
            self.lock.release()

    def get_button(self,x):
        if x==1:
            return self.draw_snake
        elif x==2:
            return self.add_volc
        elif x==3:
            return self.add_spr
        else:
            return None

    def draw_snake(self):
        if self.blck and self.blck!=1:
            self.get_button(self.blck)()
        if self.button1["relief"]=="raised":
            self.button1["relief"]="sunken"
            self.cpcid=self.canvas.mpl_connect("button_press_event",self.clickpic)
            self.canvas.get_tk_widget()["cursor"]="crosshair"
            self.lock.acquire()
            self.blck=1
        else:
            self.button1["relief"]="raised"
            if len(self.pts)>=3:
                self.snakes.add_s(self.pts)
                self.pts=[]
            self.canvas.mpl_disconnect(self.cpcid)
            self.canvas.get_tk_widget()["cursor"]="arrow"
            self.lock.release()
            self.blck=0

    def clickpic(self, event):
        if event.button==1:
            self.ax.scatter(event.xdata,event.ydata,c = 'r',marker = 'x')
            self.pts.append((event.xdata,event.ydata))
            self.canvas.draw()
        elif event.button==3:
            self.pts=[]
            self.snakes.clr()
            self.snakes.draw_pic()

    def add_volc(self):
        if self.blck and self.blck!=2:
            self.get_button(self.blck)()
        if self.button3["relief"]=="raised":
            self.button3["relief"]="sunken"
            self.cpcid=self.canvas.mpl_connect("button_press_event",self.click_volc)
            self.canvas.get_tk_widget()["cursor"]="crosshair"
            #self.lock.acquire()
            self.blck=2
        else:
            self.button3["relief"]="raised"
            self.canvas.mpl_disconnect(self.cpcid)
            self.canvas.get_tk_widget()["cursor"]="arrow"
            #self.lock.release()
            self.blck=0

    def click_volc(self, event):
        if event.button==1:
            self.snakes.volcs.append((event.xdata,event.ydata))
            self.ax.scatter(event.xdata,event.ydata,c = 'b',marker = 'o')
            self.canvas.draw()
        elif event.button==3:
            self.snakes.volcs=[]
            self.snakes.clr()
            self.snakes.draw_pic()

    def add_spr(self):
        if self.blck and self.blck!=3:
            self.get_button(self.blck)()
        if self.button4["relief"]=="raised":
            self.button4["relief"]="sunken"
            self.cpcid=self.canvas.mpl_connect("button_press_event",self.click_spr)
            self.canvas.get_tk_widget()["cursor"]="crosshair"
            #self.lock.acquire()
            self.blck=3
        else:
            self.button4["relief"]="raised"
            if len(self.spr)==4:
                self.snakes.add_spr((self.spr[0],self.spr[1]),(self.spr[2],self.spr[3]))
            self.spr=[]
            self.canvas.mpl_disconnect(self.cpcid)
            self.canvas.get_tk_widget()["cursor"]="arrow"
            #self.lock.release()
            self.blck=0

    def click_spr(self, event):
        if event.button==1:
            self.spr.append(event.xdata)
            self.spr.append(event.ydata)
            if len(self.spr)==4:
                self.snakes.add_spr((self.spr[0],self.spr[1]),(self.spr[2],self.spr[3]))
                self.spr=[]
                self.drawp=None
                self.lock.release()
            else:
                self.lock.acquire()
                self.drawp=(event.xdata,event.ydata)
            self.ax.scatter(event.xdata,event.ydata,c = 'g',marker = 'o')
            self.canvas.draw()
        elif event.button==3:
            self.snakes.sprs=[]
            self.snakes.clr()
            self.snakes.draw_pic()


if __name__=="__main__":
    win = MainWindow()
    win.start()
