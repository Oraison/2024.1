import numpy as np
import matplotlib.pyplot as plt



class perceptron:


    def __init__(self, w, b, p, data, label):
        self.w = w.T
        self.b = b
        self.p = p
        self.data = data
        self.label = label

    def update(self):
        Y = []
        for idx, (p, l) in enumerate(zip(self.data, self.label)):
            if self.activate_func(w@p+self.b) != l:
                Y.append(idx)
        
        if not Y:
            return True
        
        w_next = np.zero((2,)).T
        b_next = 0

        for idx in Y:
            w_next += label(idx)*data(x)
            b_next += label(idx)
        
        w_next = self.w + self.p * w_next
        b_next = self.b + self.p * b_next

        self.w = w_next
        self.b = b_next
        
        return False

    def activate_func(self, x):
        return 1 if x > 0 else 0



if __name__ == '__main__':
    
    fig, ax1 = plt.subplots(1,1,figsize=(9.0,8.0),sharex=True)

    ax1.set_yscale('linear')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    ax1.yaxis.grid(True,which='minor',linestyle='--')
    # ax1.legend(loc=2,prop={'size':22})

    data = np.array([[0,0], [0,1], [1,0], [1,1]])
    label = np.array([-1, 1, 1, 1])

    w = np.array([-0.5, 0.75,])
    b = 0.375

    for p, l in zip(data, label):

        ax1.scatter(*p, marker='s' if l == -1 else 'o', c = 'k' if l == -1 else 'b')

    plt.show()
