import sys
import os


import argparse
from yolov3_core import YOLO
#from PIL import Image
import glob

import cv2

def test(yolo):

    test_images = glob.glob(FLAGS.img_path+os.sep+'*.jpg')
    print('[ INFO ] Number of test images',len(test_images))
    for x in test_images:       
        print(x)
        cv_image = cv2.imread(x)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(x).replace('.jpg','.txt')
        out_image = yolo.detect_image(cv_image,filename=filename)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)       
        cv2.imwrite('testing/pred_img/'+os.path.basename(x),out_image)
       
    yolo.close_session()






FLAGS = None
if __name__ == '__main__':
    os.chdir(os.path.normpath(os.getcwd()+os.pardir+os.sep))
    print(os.getcwd())
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)  
    parser.add_argument(
        '--model_path', type=str,
        default=YOLO.get_defaults("model_path")
    )

    parser.add_argument('--test', default=False, action="store_true",help='Image detection mode, will ignore all positional arguments')  
    parser.add_argument('--img_path', default='testing', help='Image path')  

    FLAGS = parser.parse_args()

    
    test(YOLO(**vars(FLAGS)))    
 