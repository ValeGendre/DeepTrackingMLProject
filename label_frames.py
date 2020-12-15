import numpy as np
import matplotlib.pyplot as plt

name_template = 'resources/data/all_frames/vid1_{}.jpg'
bb_centers = []

#TODO Valou
id0 = 1
idmax = 146

#TODO Maga 
#id0 = 147
#idmax = 293


def creat_bbs(frame):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(frame)

    def onclick(event):
        global ix, iy
        ix, iy = int(event.xdata), int(event.ydata)
            
        bb_centers.append([ix, iy])
        fig.canvas.mpl_disconnect(cid)
        plt.close()
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

for ID in range(id0, idmax+1):
    frame = plt.imread(name_template.format(ID))
    creat_bbs(frame)
    with open('vid1_labels.txt', 'a') as f:
        f.write(name_template.format(ID)+'\n')
        f.write(str(bb_centers[-1][0]) + ' ' + str(bb_centers[-1][1]) + '\n')
