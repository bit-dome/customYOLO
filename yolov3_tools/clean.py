import glob 
import os
import argparse

def main(args):
    print('Target images folder : ',args.images)
    print('Target annotations folder : ',args.annotations)

    images_path = glob.iglob(args.images+'/*.jpg', recursive=True)
    images_name = set([os.path.basename(x).replace('.jpg','') for x in images_path])
    print('All images : ',len(images_name))


    txt_path = glob.iglob(args.annotations+'/*.txt', recursive=True)
    txt_name = set([os.path.basename(x).replace('.txt','') for x in txt_path])
    try :
        txt_name.remove('classes')
    except:
        pass
    print('All annotations : ',len(txt_name))


    print('trash images : ',len(images_name-txt_name))
    print('trash annotations : ',len(txt_name-images_name))

    #####################################################################################
    if (len(images_name-txt_name) != 0) or (len(txt_name-images_name) != 0):
        #print('\n',images_name-txt_name,txt_name-images_name)
        msg = '\nClean?'
        shall = input("%s (y/n) " % msg).lower() == 'y'

        if shall:    
            for x in images_name-txt_name:
                os.remove(args.images+'/'+x+'.jpg') 
            for x in txt_name-images_name:
                os.remove(args.annotations+'/'+x+'.txt') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trash cleanner')
    parser.add_argument('--images',default=os.getcwd(),help='Target images folder')       
    parser.add_argument('--annotations',default=os.getcwd(),help='Target annotations folder')  
    args = parser.parse_args()
    main(args)