import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class perceptron:
    def __init__(self, w, b, p, data, label):
        self.w = w.T
        self.b = b
        self.p = p
        self.data = data
        self.label = label
        self.epoch = 0
        
    def is_passed(self, point, label):
        if label < 0:
            return self.calc(point) < 0
        return self.calc(point) > 0

    def update(self):
        
        if self.epoch == 0:
            self.epoch += 1
            return
        
        Y = []
        
        for idx, (p, l) in enumerate(zip(self.data, self.label)):
            if not self.is_passed(p, l):
                Y.append(idx)
        
        if not Y:
            return np.array([])
        
        w_grad = np.zeros((2,)).T
        
        for p, l in zip(self.data[Y], self.label[Y]):
            w_grad += l*p
        
        self.b += self.p * np.sum(label[Y])
        self.w += self.p * w_grad
        
        self.epoch += 1
        
    def activate_func(self, x):
        return 1 if x > 0 else -1
    
    def get_hyper_plane_point(self, max_coord):
        y1 = -(self.w[0]*max_coord + self.b) / self.w[1]
        y2 = -(self.w[0]*-max_coord + self.b) / self.w[1]
        
        return np.array([[max_coord, y1], [-max_coord, y2]])
    
    def calc(self, point):
        return self.w @ point + self.b

if __name__ == '__main__':
    
    fig, ax1 = plt.subplots(1,1,figsize=(9.0,8.0),sharex=True)

    ax1.set_yscale('linear')
    ax1.set_xlim(-3.5, 4.5)
    ax1.set_ylim(-3.5, 4.5)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.grid(True)
    ax1.yaxis.grid(True,which='minor',linestyle='--')

    data = np.array([[0,0], [0,1], [1,0], [1,1]])
    label = np.array([-1, 1, 1, 1])
    
    w = np.array([-1., 1.])
    b = 0.
    
    # w = np.array([-0.5, 0.75,])
    # b = 0.375
    
    c1 = {'marker' : 'o', 'c' : 'b' }
    c2 = {'marker' : 's', 'c' : 'k' }
    c1_e = {'marker' : 'o', 'c' : 'r' }
    c2_e = {'marker' : 's', 'c' : 'r' }
        
    lin_reg = perceptron(w, b, 0.4, data, label)
    # hyper_line = lin_reg.get_hyper_plane_point(3)

    # # print(hyper_line)
    # ax1.text(hyper_line[0, 0], hyper_line[0, 1], lin_reg.epoch)
    # recent, = ax1.plot(hyper_line[:,0], hyper_line[:,1], c = 'maroon')

    # inital draw
    
    for p, l in zip(lin_reg.data, lin_reg.label):

        ax1.scatter(*p, **(c1 if l == 1 else c2))

    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    is_end = [False]
    hyper_lines = []
    
    def on_click_next(event):        
        if is_end[0]:
            return
        
        lin_reg.update()
        hyper_line = lin_reg.get_hyper_plane_point(3)
        
        if hyper_lines:
            hyper_lines[-1].set_color('lightsteelblue')
        
        ax1.text(hyper_line[0, 0], hyper_line[0, 1], lin_reg.epoch)
        hyper_lines.append(ax1.plot(hyper_line[:,0], hyper_line[:,1], c='maroon')[0])
        
        error_count = 0
        
        for p, l in zip(lin_reg.data, lin_reg.label):
            if not lin_reg.is_passed(p, l):
                ax1.scatter(*p, **(c1_e if l == 1 else c2_e))
                error_count += 1
            else:
                ax1.scatter(*p, **(c1 if l == 1 else c2))

        if error_count == 0:
            is_end[0] = True

        fig.canvas.draw()
        fig.canvas.flush_events()

    bnext.on_clicked(on_click_next)

    plt.show()    

