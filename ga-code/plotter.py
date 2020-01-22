import matplotlib.pyplot as plt
import time
#plt.ion()
class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 100
    def __init__(self):
        self.on_launch()
        self.xdata = []
        self.ydata = []

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], '-')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()


    def on_running(self):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    def update(self,x,y):
        self.xdata.append(x)
        self.ydata.append(y)
        self.on_running()
        time.sleep(0.5)
