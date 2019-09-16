import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback

from yolov3_tools.model import preprocess_true_boxes, yolo_loss,tiny_yolo_body,MoblrNet_body
from yolov3_tools.utils import get_data
from sklearn.utils import shuffle

import glob
import os


def read_annotations(annotation_files):
    '''Read each annotation file and convert to one list element'''
    # one file one list element
    annotations = []
    for x in annotation_files:
        with open(x) as f:
            annotations.append(f.read().replace('\n',' '))
    assert len(annotations) > 0
    return annotations
            

def _main():
    # ─── CONFIG ─────────────────────────────────────────────────────────────────────
    train_annotation_files = glob.glob('dataset/train/annotations/*.txt')   
    valid_annotation_files = glob.glob('dataset/valid/annotations/*.txt')   
    
    class_names = ['open', 'close']    
    anchors = np.array([[10,13],  [23,27],  [50,58],  [81,82],  [98,98],  [112,112]])
    num_classes = len(class_names)
    
    input_shape = (256,128) #height, width

   
    train_image_files = [x.replace('txt','jpg') for x in train_annotation_files]
    valid_image_files = [x.replace('txt','jpg') for x in valid_annotation_files]
    

    train_annotations = read_annotations(train_annotation_files)
    valid_annotations = read_annotations(valid_annotation_files)

        
    # ─── CREATE TRAIN GRAPH WITH MODEL AND LOSS CALC ────────────────────────────────
    model = create_MoblrNet_model((input_shape[0],input_shape[1],3), anchors, num_classes,load_pretrained=True,weights_path="g1-06.h5")

    # Compile model
    model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})


    # Training
    batch_size = 8
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_annotations), len(valid_annotations), batch_size))
    
    checkpoint = ModelCheckpoint("g1-{epoch:02d}.h5", monitor='val_loss', verbose=1, save_best_only=False, mode='max',save_weights_only=False)
    history = model.fit_generator(data_generator(train_annotations,train_image_files, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, len(train_annotations)//batch_size),
            validation_data=data_generator(valid_annotations,valid_image_files, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, len(valid_annotations)//batch_size),
            epochs=50,
            initial_epoch=0,
            verbose=1,
            callbacks=[checkpoint])
    

   # ────────────────────────────────────────────────────────────────────────────────


def create_tiny_model(input_shape, anchors, num_classes,load_pretrained=False):
    '''create the training model, for Tiny YOLOv3'''
    weights_path = "tiny.h5"
    K.clear_session() # get a new session
    image_input = Input(shape=(None,None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
            model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(weights_path))

            freeze_body = False
            if freeze_body in [1, 2]:
                # Freeze the darknet body or freeze all but 2 output layers.
                num = (20, len(model_body.layers)-2)[freeze_body-1]
                for i in range(num): model_body.layers[i].trainable = False
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
   

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    model.summary()

    return model

def create_MoblrNet_model(input_shape, anchors, num_classes,load_pretrained=False,weights_path=None):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    h, w, _ = input_shape
    
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = MoblrNet_body(input_shape, num_anchors//2, num_classes)
    print('Create MoblrNet YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path)
        print('Load weights {}.'.format(weights_path))

   

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    model.summary()

    return model


def data_generator(annotation_lines, im_names,batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        annotation_lines,im_names=shuffle(annotation_lines,im_names,random_state=0)
        for b in range(batch_size):
            # Get data of one image
            image, box = get_data(annotation_lines[i],im_names[i], input_shape, random=True)            
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


if __name__ == '__main__':
    _main()







