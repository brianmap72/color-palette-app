from keras.models import load_model
import numpy as np
from PIL import Image
import keras
import itertools

generator = load_model('epoch_10.hdf5')
print(generator)

colorlist = []

input_color = [( (66, 105,  78), (183, 166, 119), (62, 97, 116), (157, 163, 49), (116, 100, 63))]

for i, color in enumerate(input_color):
    print('Processing {} ... '.format(str(i)))
    print(color)

    batch = int(256 / 5)
    for num in range(5):
        print(num)
        img = Image.new("RGB",(256,1),(0,0,0))
        im = np.array(img)
        im_fake = np.array(img)

        index = [num]
        print(index)
        for j, c in enumerate(color):
            if j not in index:
                if j == len(color)-1:
                    im_fake[:,j*batch:,:] = c
                else:
                    im_fake[:,j*batch:(j+1)*batch,:] = c

            if j == len(color)-1:
                im[:,j*batch:,:] = c
            else:
                im[:,j*batch:(j+1)*batch,:] = c

        result = generator.predict(np.expand_dims(im_fake / 255 * 2 - 1,0))[0]
        result[:,0,:] = np.mean(result[:,0:(1)*batch,:],1)[0]

        if num==0:
            newcolor = (((result[0,0,:]+1)/2*255).astype('uint8'))
        elif  num==1:
            newcolor = (((result[0,51,:]+1)/2*255).astype('uint8'))
        elif  num==2:
            newcolor = (((result[0,102,:]+1)/2*255).astype('uint8'))
        elif  num==3:
            newcolor = (((result[0,153,:]+1)/2*255).astype('uint8'))
        else:
            newcolor = (((result[0,204,:]+1)/2*255).astype('uint8'))
                        
        print(newcolor)
        colorlist.append(newcolor.tolist())

    print("colorlist", colorlist)
    merged = list(itertools.chain(*colorlist))
    print("merged", merged)

