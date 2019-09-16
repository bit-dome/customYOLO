import sys, os, time

from timeit import default_timer as timer
import cv2
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *

from yolov3_tools.model import yolo_eval, yolo_body, MoblrNet_body
from yolov3_tools.utils import letterbox_image

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import InputLayer

from PIL import Image, ImageFont, ImageDraw

os.chdir(sys.path[0])

class YOLO(object):
    _defaults = {
        "model_path": 'g1-06.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/classes.txt',
        "score" : 0.3,
        "iou" : 0.1,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
        "test":False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)    

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)   
      
        self.yolo_model = MoblrNet_body(Input(shape=(416,416,3)), num_anchors//2, num_classes)
        self.yolo_model.load_weights(self.model_path)
      

        #plot_model(self.yolo_model, to_file='model.png',show_shapes=1)

        print('{} model, anchors, and classes loaded.'.format(model_path))     
        # box colors
        self.colors = {"open":(0,255,0),"close":(255,0,0)}
        
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))       
        boxes, scores, classes = yolo_eval(self.yolo_model.output,
                                           self.anchors,
                                           len(self.class_names), 
                                           self.input_image_shape,
                                           score_threshold=self.score, 
                                           iou_threshold=self.iou)
 
        #self.yolo_model.summary()
        return boxes, scores, classes

    def detect_image(self, image,filename='',boxes_only=False):
        
       
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))       
        image_data = np.array(boxed_image, dtype='float32')        
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0) 

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })     

      
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
        
           
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            
           
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[predicted_class], 2)            
            cv2.putText(image,label, (left,top+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[predicted_class])
        

                 
         
        return image

    def close_session(self):
        self.sess.close()


