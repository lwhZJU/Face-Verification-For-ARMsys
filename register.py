#-*- encoding: utf-8 -*-
import sys
sys.path.append("/home/binhanxu/下载/caffe/python/caffe")
import caffe
import cv2
import numpy as np
import time
from demo import detect_face
import copy
import os

"""Calculate average distance of vectors"""
def dist_ave(vecs):
    dist_all = 0
    dist_ave0 = 10000
    num = 0
    for i in range(len(vecs) - 1):
        for j in range(i + 1, len(vecs)):
            dist_all += distance(vecs[i], vecs[j])
            num += 1
    if(num!=0):
        dist_ave0 = dist_all / num
    return dist_ave0

"""L2 distance"""
def distance(vector1,vector2):
    return np.linalg.norm(vector1 - vector2)

"""Menhaton distance"""
def menhaton(vector1,vector2):
    return sum(abs(vector1 - vector2))

"""Cos distance"""
def cos(vector1,vector2):
    cosV12 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosV12

"""Pretreatment before input squeezenet"""
def dealpic(im):
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    return in_

"""Get result"""
def cal_vector(input):
    net.blobs['data'].reshape(1, *input.shape)
    net.blobs['data'].data[...] = input
    net.forward()
    out = net.blobs['pool10'].data[0]
    # out = net.blobs['prob'].data[0].argmax(axis=0)
    return out

"""From image to result"""
def get_vec(im):
    return cal_vector(dealpic(im))

"""Find nereast vector of vec0 in vecs"""
def find(vec0, vecs=[]):
    dist_save = []
    for vec in vecs:
        dist_save.append(distance(vec0, vec))
    dist = min(dist_save)
    num = dist_save.index(dist)
    return dist, num

if __name__ == '__main__':
    root_dir = os.getcwd().replace('\\', '/')
    cap = cv2.VideoCapture(0)
    caffe.set_mode_cpu()
    caffe_model_path = "./model"

    """The verification caffenet"""
    # You can use 96 to replace 64
    net = caffe.Net(caffe_model_path +"/crop64.prototxt",
                    caffe_model_path+"/crop64.caffemodel", caffe.TEST)
    crop_size = 64 #Set the input size
    """The detection caffenet"""
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    PNet = caffe.Net(caffe_model_path + "/det1.prototxt", caffe_model_path + "/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path + "/det2.prototxt", caffe_model_path + "/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path + "/det3.prototxt", caffe_model_path + "/det3.caffemodel", caffe.TEST)

    """Variables init"""
    # For cv2
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font1 = cv2.FONT_HERSHEY_PLAIN

    # Store vectors,names,thresholds.
    vecs_save = [] 
    vecs=[] 
    names=[]
    # Store by people
    vec_people = []
    name_people = []
    thre_people = []

    index=0 #No use
    threshold0=10000

    # Record time
    time_all=[]
    time_10=[]
    frame_num = 0 # Frame number of complete detection and verification

    boundingboxes=[] #MTCNN detection boxes

    scale = 5  # Zoom for accelerate detection
    padding=0.15 # Detection paddnig ratio
    detect_flag  = 0 # For tailor detection area
    count_flag_0 = 0 # For tailor detection area
    count_flag_1 = 0 # For tailor detection area
    load_flag = 0 # If load pictures
    thre_flag = 0 # If load and use threshold
    thres_low = 1.8 # Threshold of same
    thres_high = 1.8 # Threshold of undefined
    anchor_x = [] # For one key register
    anchor_name = [] # For one key register

    n_users = 3 # Max of users to detect in the mean time
    
    # 循环检测识别人脸
    while True:
        stage_flag=0
        _, frame = cap.read()
        # Start time
        all_time_s = time.time()
        detect_time_s = time.time()
        # BGR -> RGB
        frame_c = frame.copy()
        frame_c[:, :, 0] = frame[:, :, 2]
        frame_c[:, :, 2] = frame[:, :, 0]
        # Small size for detection
        img = cv2.resize(frame_c, ((int(frame.shape[1]/scale), int(frame.shape[0]/scale))))
        img_c = img.copy()
        img_input = img_c
        """############## Start detection ###################"""
        # full picture
        if detect_flag == 0:
            count_flag_0 += 1
            img_input = img_c  ###
            boundingboxes, points = detect_face(img_input, minsize, PNet, RNet, ONet, threshold, False, factor)
            if len(boundingboxes) > 0:
                detect_flag = 1
        #3 users,shaped picture
        elif detect_flag == 1:
            count_flag_1 += 1
            img_input = img_Yshaped
            boundingboxes, points = detect_face(img_input, minsize, PNet, RNet, ONet, threshold, False, factor)
            for i in range(len(boundingboxes)):
                boundingboxes[i][1] += y_upbound
                boundingboxes[i][3] += y_upbound
            if len(boundingboxes) == 2:
                if count_flag_1 % 40 == 0:
                    detect_flag = 0
            if len(boundingboxes) == 1:
                if count_flag_1 % 40 == 0:
                    detect_flag = 0
            if len(boundingboxes) == 0:
                detect_flag = 0
        # Finish detection, print time
        detect_time_e = time.time()
        # print("detect time:",detect_time_e-detect_time_s)
        """############## End of detection ###################"""

        """Coordinate transformation"""
        if len(boundingboxes) != 0:
            y0_list = [int(box[1]) for box in boundingboxes]
            y1_list = [int(box[3]) for box in boundingboxes]

            y_upbound = max(min(y0_list), 0)
            y_lowbound = max(y1_list)
            y_h=y_lowbound-y_upbound
            y_upbound = max(min(y0_list)-y_h*0.2, 0)
            y_lowbound = max(y1_list)+y_h*0.2
            img_Yshaped = img_c[int(y_upbound):int(y_lowbound), :]

        if(len(boundingboxes)):
            boundingboxes_old=boundingboxes

        xm00_save=[]
        ym00_save=[]
        img0_save=[]

        for i in range(len(boundingboxes)):
            x0=(max(boundingboxes[i][0],0))
            y0=(max(boundingboxes[i][1],0))
            x1=(max(boundingboxes[i][2],0))
            y1=(max(boundingboxes[i][3],0))

            # Coordinate from img to frame
            [x00,y00,x10,y10]=[int(x0*scale),int(y0*scale),int(x1*scale),int(y1*scale)]

            if(y1-y0>10)and(x1-x0>10)and(img.shape[0]>10):
                h00 = y10 - y00
                w00 = x10 - x00
                xm00 = (x00 + x10) / 2
                ym00 = (y00 + y10) / 2
                xm00_save.append(xm00)
                ym00_save.append(ym00)
                # Show x_ave coordinate
                cv2.putText(frame, "x:"+str(xm00), (x10 - 20, y10 - 20), font1, 2, (0, 0, 255), 2)
                padding_left = x00 - padding * w00
                padding_right = frame_c.shape[1]- (x10 + padding * w00)
                padding_top = y00 - padding * h00
                padding_bottom = frame_c.shape[0] - (y10 + padding * h00)
                # Face out of range,quit detection
                if padding_left<10 or padding_right<0 or padding_bottom<0 or padding_top<10:
                    continue
                # Crop and resize poi from frame
                img0 = cv2.resize(frame_c[int(max(0,padding_top)):int(min(frame_c.shape[0],y10 + padding * h00)),int(max(0,padding_left)):int(min(frame_c.shape[1],x10 + padding * w00))], (crop_size, crop_size))
                # Save cropped faces
                img0_save.append(img0)
                cv2.rectangle(frame, (x00, y00),
                              (x10, y10), (0, 255, 0), 1)

                """############## Start verification ###################"""
                if(len(vecs)>0):
                    verify_time_s = time.time()
                    # Get the output vector from caffenet
                    vec0 = get_vec(img0)
                    stage_flag=1
                    frame_num+=1

                    vecs=copy.deepcopy(vecs_save)
                    dist,num = find(vec0,vecs)
                    if(len(name_people)>0)and(thre_flag==1):
                        find_name=names[num]
                        find_index=name_people.index(find_name)
                        threshold0=thre_people[find_index]
                    vecs_save = copy.deepcopy(vecs)

                    # Same
                    if(dist<int(threshold0*thres_low)):
                        cv2.putText(frame, names[num], (x00 + 20, y00 + 20), font, 2, (255, 255, 0),2)
                    # Undefined
                    elif(dist>int(threshold0*thres_high)):
                        cv2.putText(frame, "undef", (x00 + 20, y00 + 20), font, 2, (255, 255, 0), 2)
                    verify_time_e = time.time()
                    # print("verify time:", verify_time_e - verify_time_s)
            """############## End of verification ###################"""

        if(stage_flag):
            all_time_e = time.time()
            inference = all_time_e - all_time_s
            # print("Whole time:", inference)  # Inference once
            time_10.append(inference)
            if(frame_num % 10==0):
                inference_10= sum(time_10)/len(time_10)
                time_all.append(inference_10)
                time_10=[]
                print("Frame num:", frame_num)
                print("Average time:", inference_10) # 10 frames average inference
                print("*********************")

        cv2.imshow("人脸识别", frame)
        k = cv2.waitKey(10)

        """##############  Define key functions ###################"""

        """ Define key functions
        r->Register faces manually ; l->Load pictures and get vectors ; s->Register faces automatically
        c->Calculate thresholds; t->Set threshold ratio; e->Quit and print average inference of whole process
       """

        if k & 0xFF == ord('r'):
            thre_flag = 0
            anchor_x=[]
            anchor_name=[]
            useful=len(xm00_save)
            print("Number of faces:",useful)
            for i in range(useful):
                print("No."+str(i)+": x="+str(xm00_save[i]))
            end_flag=0
            for i in range(useful):
                if(end_flag==1):
                    break;
                print("Select number:")
                input_str=input()
                if(input_str=='end'):
                    end_flag=1
                    continue;
                else:
                    num=int(input_str)
                    img0=img0_save[num]
                    print("Input name:")
                    input_str = input()
                    if (input_str == 'end'):
                        continue;
                    else:
                        name0=input_str

                        pic_names=os.listdir('./capture')
                        print(pic_names)
                        name_match=0
                        for name in pic_names:
                            if(name0==name):
                                pic_num=len(os.listdir('./capture'+'/'+name0))
                                cv2.imwrite('./capture/' + str(name0) + '/' +str(pic_num)+'.jpg',img0)
                                name_match=1
                                break;
                        if(len(pic_names)==0 or name_match==0):
                            path_save = './capture/' + str(name0)
                            os.makedirs(path_save)
                            cv2.imwrite(path_save + '/0.jpg', img0)

                    vec0=get_vec(img0)
                    vecs=copy.deepcopy(vecs_save)
                    vecs.append(vec0)
                    vecs_save=copy.deepcopy(vecs)
                    anchor_name.append(name0)
                    anchor_x.append(xm00_save[num])
                    names.append(name0)
                    index+=1

        # load
        if k & 0xFF == ord('l'):
            names=[]
            vecs=[]
            load_flag=1
            thre_flag = 1
            pic_names = os.listdir('./capture')
            for name in pic_names:
                load_path='./capture/'+name
                pics=os.listdir(load_path)
                for picture in pics:
                    img0=cv2.resize(cv2.imread(load_path+'/'+picture),(crop_size,crop_size))
                    names.append(str(name))
                    vecs=copy.deepcopy(vecs_save)
                    vecs.append(get_vec(img0))
                    vecs_save=copy.deepcopy(vecs)
            # print(names)

        if k & 0xFF == ord('s'):
            if(len(boundingboxes)==3 and len(anchor_name)==3):
                anchor_adresss = ['./capture/%s' % anchor_name[0], './capture/%s' % anchor_name[1],
                                  './capture/%s' % anchor_name[2]]
                anchor_numbers = [len(os.listdir(anchor_adresss[0])), len(os.listdir(anchor_adresss[1])),
                                  len(os.listdir(anchor_adresss[2]))]
                x_imgs = xm00_save
                imgs_list = img0_save
                for i in range(len(x_imgs)):
                    for j in range(len(anchor_x)):
                        if abs(x_imgs[i] - anchor_x[j]) < 50:
                            anchor_x[j] = x_imgs[i]
                            address_see=anchor_adresss[j] + '/' +str(anchor_numbers[j])+'.jpg'
                            print(address_see)
                            cv2.imwrite(address_see , imgs_list[i])

        if k & 0xFF == ord('c'):
            vec_people=[]
            name_people=[]
            thre_people=[]
            vec_i=[]
            if(len(names)):
                name_old=names[0]
                i=0
                for name in names:
                    if(name!=name_old):
                        vec_people.append(vec_i)
                        vec_i=[]
                        name_people.append(name_old)
                    vec_i.append(vecs[i])
                    i += 1
                    name_old=name
                vec_people.append(vec_i)
                vec_i = []
                name_people.append(name_old)

            for vecs in vec_people:
                if(len(vecs)>0):
                    thre_people.append(dist_ave(vecs))
            print(name_people)
            print(thre_people)
            # print(len(vecs))


        if k & 0xFF == ord('t'):
            print("low:")
            thres_low=input()
            print("high")
            thres_high=input()
            print(thres_low,thres_high)
            
        if k & 0xFF == ord('e'):
            inference_all=sum(time_all)/len(time_all)
            print("*************************")
            print("Final time:",inference_all)
            print("*************************")
            break

    cap.release()
    cv2.destroyAllWindows()
