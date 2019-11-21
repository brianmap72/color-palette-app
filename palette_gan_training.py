from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
import keras.backend as K
import os
from PIL import Image, ImageDraw
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(allow_growth=True) #tf.GPUOptions(allow_growth=allow_growth)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization


def UNET_G(isize, nc_in=3, nc_out=3, ngf=128, fixed_input_size=True):    
    max_nf = 8*ngf    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=(1,4), strides=(1,2), use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=-1)([x, x2])            
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=(1,4), strides=(1,2), use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,          
                            name = 'convt.{0}'.format(s))(x)        
        x = Cropping2D(cropping=((0,0),(1,1)))(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        return x
    
    s = isize if fixed_input_size else None

    _ = inputs = Input(shape=(1, s, nc_in))        
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])

# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)
def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """    

    input_a, input_b = Input(shape=(None, None, nc_in)), Input(shape=(None, None, nc_out))
    _ = Concatenate(axis=-1)([input_a, input_b])
    _ = conv2d(ndf, kernel_size=(1,4), strides=(1,2), padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=(1,4), strides=(1,2), padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D((0,1))(_)
    _ = conv2d(out_feat, kernel_size=(1,4),  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    
    # final layer
    _ = ZeroPadding2D((0,1))(_)
    _ = conv2d(1, kernel_size=(1,4), name = 'final'.format(out_feat, 1), 
               activation = "sigmoid") (_)    
    return Model(inputs=[input_a, input_b], outputs=_)


nc_in = 3
nc_out = 3
ngf = 128
ndf = 128
λ = 10

imageSize = 256
batchSize = 64
lrD = 2e-4
lrG = 2e-4

# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

netD = BASIC_D(nc_in, nc_out, ndf)
netG = UNET_G(imageSize, nc_in, nc_out, ngf)

real_A = netG.input
fake_B = netG.output
netG_generate = K.function([real_A], [fake_B])
real_B = netD.inputs[1]
output_D_real = netD([real_A, real_B])
output_D_fake = netD([real_A, fake_B])

from keras.optimizers import RMSprop, SGD, Adam

#loss_fn = lambda output, target : K.mean(K.binary_crossentropy(output, target))
loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

loss_D_real = loss_fn(output_D_real, K.ones_like(output_D_real))
loss_D_fake = loss_fn(output_D_fake, K.zeros_like(output_D_fake))
loss_G_fake = loss_fn(output_D_fake, K.ones_like(output_D_fake))

loss_L1 = K.mean(K.abs(fake_B-real_B))

loss_D = loss_D_real +loss_D_fake
training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(netD.trainable_weights,[],loss_D)
netD_train = K.function([real_A, real_B],[loss_D/2], training_updates)

loss_G = loss_G_fake   + 100 * loss_L1
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(netG.trainable_weights,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_G_fake, loss_L1], training_updates)

def load_data(dataPath):
    dataAB = []
    for root,dirs,files in os.walk(dataPath):
        for f in files:
            if f.endswith('jpg'):
                dataAB.append(os.path.join(root,f))
    random.shuffle(dataAB)
    
    return dataAB[:int(len(dataAB)*0.85)], dataAB[int(len(dataAB)*0.85):]

def read_image(img_name):
    im = Image.open(img_name)
    im_fake = im.copy()
    draw_fake = ImageDraw.Draw(im_fake)
    batch = int(im.size[0] / 5)
    nb = random.randint(1,4)
    index = random.sample([k for k in range(5)],nb)

    for j in range(5):
        if j in index:
            if j != 4:
                draw_fake.rectangle((j*batch,0,(j+1)*batch,1),fill=(0,0,0))
            else:
                draw_fake.rectangle((j*batch,0,im.size[0],1),fill=(0,0,0))

    imgB = im
    imgA = im_fake

    outA = np.array(imgA.resize((256,1))) / 255 * 2 - 1
    outB = np.array(imgB.resize((256,1))) / 255 * 2 - 1
    return outA, outB


def minibatch(dataAB, batchsize, direction=0):
    length = len(dataAB) * 10
    epoch = i = 0
    tmpsize = None    
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            random.shuffle(dataAB)
            i = 0
            epoch+=1        
        dataA = []
        dataB = []
        for j in range(i,i+size):
            imgA,imgB = read_image(dataAB[int(j % len(dataAB))])
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i+=size
        tmpsize = yield epoch, dataA, dataB  


def showX(X, epoch=0, rows=3):

    plt.figure(figsize=(10,5))
    X = np.concatenate(X,1)
    for i in range(rows):
        plt.subplot(2,3,i+1)
        plt.imshow(np.array(Image.fromarray(((X[i]+1)/2*255).clip(0,255).astype('uint8')).resize((512,100))))
    plt.show()
    plt.savefig('results/epoch_{}.jpg'.format(str(epoch)))

trainAB, valAB = load_data('./peisenet_palettes/full_colorlover/')
train_batch = minibatch(trainAB, 6)
_, trainA, trainB = next(train_batch)

showX([trainA,trainB])
_,a,b=train_batch.send(6)
del train_batch, trainA, trainB

def netG_gen(A):
    return np.concatenate([netG_generate([A[i:i+1]])[0] for i in range(A.shape[0])], axis=0)

import time
t0 = time.time()
niter = 10
gen_iterations = 0
errL1 = epoch = errG = 0
errL1_sum = errG_sum = errD_sum = 0

display_iters = 2000
val_batch = minibatch(valAB, 6)
train_batch = minibatch(trainAB, batchSize)
current_iter = 0

while epoch < niter: 
    epoch, trainA, trainB = next(train_batch)        
    errD,  = netD_train([trainA, trainB])
    errD_sum +=errD

    print(epoch)

    errG, errL1 = netG_train([trainA, trainB])
    errG_sum += errG
    errL1_sum += errL1
    
    for i in range(3):        
        print("i ", i)
        _, trainA1, trainB1 = next(train_batch)
        errG, errL1 = netG_train([trainA1, trainB1])
        errG_sum += errG
        errL1_sum += errL1
    
    print("gen_iterations", gen_iterations)
    print("current_iter", current_iter)

    gen_iterations+=1
    if gen_iterations%display_iters==0:
        # if gen_iterations%(5*display_iters)==0:
            # clear_output()
        print('[%d/%d][%d] Loss_D: %f Loss_G: %f loss_L1: %f' % (epoch, niter, gen_iterations, errD_sum/display_iters, errG_sum/(display_iters*4), errL1_sum/(display_iters*4)))
        _, valA, valB = train_batch.send(6) 
        fakeB = netG_gen(valA)
        showX([valA, valB, fakeB], gen_iterations)
        errL1_sum = errG_sum = errD_sum = 0
        _, valA, valB = next(val_batch)
        fakeB = netG_gen(valA)
        showX([valA, valB, fakeB], gen_iterations)
    if epoch > current_iter:
        current_iter += 1
        netG.save('weights/100L1_128channel_4gen/generator/epoch_{}.hdf5'.format(str(epoch)))
        netD.save('weights/100L1_128channel_4gen/discriminator/epoch_{}.hdf5'.format(str(epoch)))
