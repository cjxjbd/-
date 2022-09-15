import os
import cv2
import scipy.io as scio
import numpy as np
from face_detection import RetinaFace

def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w/3
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

if __name__=='__main__':
    # img_path='/home/ray/PycharmProjects/L2CS-Net-main/MPIIFaceGaze/p02/day02/0037.jpg'
    # lab='day02/0037.jpg 158 557 549 386 594 382 672 379 720 379 590 512 684 508 -0.144698 0.060018 -0.036000 -9.202821 13.498850 533.204169 -8.482285 36.110225 529.885237 107.413639 107.679300 20.951624 left'
    # c_lab='p02/face/1123.jpg p02/left/1123.jpg p02/right/1123.jpg day02/0037.jpg left 0.17622093374243358,0.18625829429281845,-0.9665681715833988 -0.07834310529204966,0.07571605470898865,0.002968857464359205 -0.18033543662859822,-0.18735241284630438 0.07575484944686207,0.0783806008440235 0.067743205,0.017196741,0.03575279 1.0,1.0,1.1295564631728443 6.2266485656437e-07,-2.011064054840972e-06,600.0000293385957'

    name='0026.jpg'
    impath='/home/zyp/pythonProject/L2CS-Net-main/MPIIFaceGaze/p03'
    labtxt='/home/zyp/pythonProject/L2CS-Net-main/MPIIFaceGaze/p03/p03.txt'
    c_labtxt='/home/zyp/pythonProject/L2CS-Net-main/datasets/MPIIFaceGaze/Label/p03.label'

    with open(labtxt,'r') as l_t:
        line_a=l_t.readline()
        while line_a:
            #print(line_a.split(' ')[0])
            if line_a.split(' ')[0] == name:
                lab=line_a
                img_path=os.path.join(impath,name)
                print(line_a)
                break
            line_a = l_t.readline()

    with open(c_labtxt, 'r') as cl_t:
        line_a = cl_t.readline()
        while line_a:
            if line_a.split(' ')[3] == name:
                c_lab = line_a
                print(line_a)
                break
            line_a = cl_t.readline()

    screen_mat='/home/zyp/pythonProject/L2CS-Net-main/MPIIFaceGaze/p02/Calibration/Camera.mat'
    line = c_lab.strip().split(" ")
    gaze2d = line[7]
    gaz = np.array(gaze2d.split(",")).astype("float")
    # pitch = gaz[0]* 180 / np.pi
    # yaw = gaz[1]* 180 / np.pi
    pitch = gaz[0]
    yaw = gaz[1]

    print('line',line)

    s_mat=scio.loadmat(screen_mat)
    #print('s_mat',s_mat)

    lab=lab.split(sep=' ')
    print('label lenth:',len(lab))
    print('label data:',lab)

    img=cv2.imread(img_path)
    print('img.shape:',img.shape)
    detector = RetinaFace(gpu_id=0)
    faces = detector(img)
    for box, landmarks, score in faces:
        x_min = int(box[0])
        y_min = int(box[1])
        bbox_width = int(box[2]) - x_min
        bbox_height = int(box[3])- y_min
        face=img[y_min:int(box[3]), x_min:int(box[2])]


    point_size = 1
    point_color = (0, 0, 255)
    thickness = 4

    coors={}
    # coors={key1:[x1,y1],key2:[x2,y2],...}
    coors['g_loc']=[lab[1],lab[2]]

    for i in range(6):
        coors['lm_'+str(i)]=[lab[3+2*i],lab[3+2*i+1]]

    for key,coor in coors.items():
        cv2.circle(img, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)
        cv2.putText(img, key, (int(coor[0]), int(coor[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.putText(img, lab[-1], (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    draw_gaze(x_min,y_min,bbox_width, bbox_height,img,(pitch,yaw),color=(0,0,255))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

