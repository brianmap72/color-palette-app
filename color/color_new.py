from PIL import Image, ImageDraw
from webcolors import rgb_to_hex
import colorgram
import cv2
from colorsys import rgb_to_hsv, hsv_to_rgb
from colorharmonies import Color, complementaryColor, triadicColor, splitComplementaryColor, tetradicColor, analogousColor, monochromaticColor
import pandas as pd
import numpy as np
from top_color import *
from tensorflow.python.keras.models import load_model
import keras
import itertools
import tensorflow as tf

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(infile, outline_width, palette_length_div, outline_color, numcolors=5):
    original_image = Image.open(infile)
    image = Image.open(infile)
    small_image = image.resize((80, 80))
    result = small_image.convert('RGB', palette=Image.ADAPTIVE, colors=numcolors)   # image with only 10 dominating colors
    result.putalpha(0)

    colors = colorgram.extract(small_image, numcolors)
    input_color = np.array([x.rgb for x in colors]).flatten()
    input_color = [(int(input_color[i]), int(input_color[i+1]), int(input_color[i+2])) for i in [0,3,6,9,12]]
    generator = load_model('epoch_10.hdf5')
    colorlist = []

    batch = int(256 / 5)
    for num in range(5):
        print(num)
        img = Image.new("RGB",(256,1),(0,0,0))
        im = np.array(img)
        im_fake = np.array(img)

        index = [num]
        print(index)
        for j, c in enumerate(input_color):
            if j not in index:
                if j == len(input_color)-1:
                    im_fake[:,j*batch:,:] = c
                else:
                    im_fake[:,j*batch:(j+1)*batch,:] = c

            if j == len(input_color)-1:
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
        
        colorlist.append(newcolor.tolist())

    output_img = list(itertools.chain(*colorlist))
    output_color = [(int(output_img[i]), int(output_img[i+1]), int(output_img[i+2])) for i in [0,3,6,9,12]]

    swatchsize2 = 100
    width, height = original_image.size
    palette_height = int(height/palette_length_div)
    background = Image.new("RGB", (width, height))   # blank canvas(original image + palette)
    pal2 = Image.new("RGB", (numcolors*swatchsize2, swatchsize2))
    pal3 = Image.new("RGB", (numcolors*swatchsize2, swatchsize2))
    pal4a = Image.new("RGB", (numcolors*swatchsize2, swatchsize2))
    pal4b = Image.new("RGB", (numcolors*swatchsize2, swatchsize2))
    pal5a = Image.new("RGB", (numcolors*swatchsize2, swatchsize2))
    pal5b = Image.new("RGB", (numcolors*swatchsize2, swatchsize2))
    palml = Image.new("RGB", (numcolors*swatchsize2, swatchsize2))

    draw1 = ImageDraw.Draw(pal2)
    draw2 = ImageDraw.Draw(pal3)
    draw3a = ImageDraw.Draw(pal4a)
    draw3b = ImageDraw.Draw(pal4b)
    draw4a = ImageDraw.Draw(pal5a)
    draw4b = ImageDraw.Draw(pal5b)
    drawml = ImageDraw.Draw(palml)

    posx1 = 0
    posx2 = 0
    posx3a = 0
    posx3b = 0
    posx4a = 0
    posx4b = 0
    posxml = 0

    swatchsize = width/10
    hex_codes = []
    hex_codes_compli = []
    hex_codes_triadic1 = []
    hex_codes_triadic2 = []
    hex_codes_tetradic1 = []
    hex_codes_tetradic2 = []
    hex_codes_ml = []

    for color in colors:

        draw1.rectangle([posx1, 0, posx1+swatchsize2, swatchsize2], fill=color.rgb)
        posx1 = posx1 + swatchsize2
        hex_codes.append(RGB2HEX(color.rgb))

    for color in output_color:

        drawml.rectangle([posxml, 0, posxml+swatchsize2, swatchsize2], fill=color)
        posxml = posxml + swatchsize2
        hex_codes_ml.append(RGB2HEX(color))
    
    for color in colors:

        colorconv = Color([color.rgb.r, color.rgb.g, color.rgb.b],"","")
        ret_complicolor = complementaryColor(colorconv)
        compcolor = "#{:02x}{:02x}{:02x}".format(int(ret_complicolor[0]), int(ret_complicolor[1]), int(ret_complicolor[2]))
        draw2.rectangle([posx2, 0, posx2+swatchsize2, swatchsize2], fill=(compcolor))

        posx2 = posx2 + swatchsize2
        hex_codes_compli.append(RGB2HEX(ret_complicolor))

    for color in colors:

        colorconv = Color([color.rgb.r, color.rgb.g, color.rgb.b],"","")
        ret_triadiccolor = triadicColor(colorconv)
        print(ret_triadiccolor)
        tricolora1 = "#{:02x}{:02x}{:02x}".format((ret_triadiccolor[0][0]), (ret_triadiccolor[0][1]), (ret_triadiccolor[0][2]))
        tricolora2 = "#{:02x}{:02x}{:02x}".format((ret_triadiccolor[1][0]), (ret_triadiccolor[1][1]), (ret_triadiccolor[1][2]))

        draw3a.rectangle([posx3a, 0, posx3a+swatchsize2, swatchsize2], fill=(tricolora1))
        draw3b.rectangle([posx3b, 0, posx3b+swatchsize2, swatchsize2], fill=(tricolora2))

        posx3a = posx3a + swatchsize2
        posx3b = posx3b + swatchsize2
        hex_codes_triadic1.append(RGB2HEX(ret_triadiccolor[0]))
        hex_codes_triadic2.append(RGB2HEX(ret_triadiccolor[1]))

    for color in colors:

        colorconv = Color([color.rgb.r, color.rgb.g, color.rgb.b],"","")
        ret_tetracolor = analogousColor(colorconv)
        tetracolora1 = "#{:02x}{:02x}{:02x}".format((ret_tetracolor[0][0]), (ret_tetracolor[0][1]), (ret_tetracolor[0][2]))
        tetracolora2 = "#{:02x}{:02x}{:02x}".format((ret_tetracolor[1][0]), (ret_tetracolor[1][1]), (ret_tetracolor[1][2]))

        draw4a.rectangle([posx4a, 0, posx4a+swatchsize2, swatchsize2], fill=(tetracolora1))
        draw4b.rectangle([posx4b, 0, posx4b+swatchsize2, swatchsize2], fill=(tetracolora2))

        posx4a = posx4a + swatchsize2
        posx4b = posx4b + swatchsize2
        hex_codes_tetradic1.append(RGB2HEX(ret_tetracolor[0]))
        hex_codes_tetradic2.append(RGB2HEX(ret_tetracolor[1]))

    del draw1
    del draw2
    del draw3a
    del draw3b
    del draw4a
    del draw4b
    del drawml

    box = (0, height, width, height)

    # pasting image and palette on the canvas
    background.paste(original_image)


    return background, pal2, hex_codes, pal3, hex_codes_compli, pal4a, hex_codes_triadic1, pal4b, hex_codes_triadic2, pal5a, hex_codes_tetradic1, pal5b, hex_codes_tetradic2, palml, hex_codes_ml
