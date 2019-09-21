# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:54:12 2019

@author: lenovo
"""
import math
import numpy as np
import random

def random_attributes(n,min_size,max_size,angle,color,shape,Location):
    
        num=0
        file = open('zm.txt','w') 
        while num < n :
            num  += 1
            size  = random.uniform(min_size, max_size)
            angle = random.uniform(-math.pi,math.pi)
            color = []
            for i in range(3):
                a = random.randint(0,255)
                color.append(a)
            all_shape = ['rectangle','circle','triangle','ellipse','parallelogram','trapezium']
            shape = random.choice(all_shape)
            location = []
            for i in range(4):
                b = random.uniform(0,224)
                b = float('%.2f'%b)
                location.append(b)    
                
            
            file.write('picture:{}  Shape: {}      Angle:{:.2f}    Location:{}     Color:{}    Size:{:.2f}\n'.format(num,shape,angle,location,color,size))
        return shape,angle,location,color,size,min_size,max_size,
        file.close()
#%%定义不同图形的画法
from skimage.draw import polygon as draw_polygon, circle as draw_circle,ellipse as draw_ellipse
from skimage._shared.utils import warn


def _generate_rectangle_mask(point, image, rshape, random,angle=0):
    available_width = min(image[1] - point[1], rshape[1])
    if available_width < rshape[0]:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image[0] - point[0], rshape[1])
    if available_height < rshape[0]:
        raise ArithmeticError('cannot fit shape to image')
        
    # Pick random widths and heights.
    r = random.randint(rshape[0], available_height + 1)
    c = random.randint(rshape[0], available_width + 1)
    
    p1r=point[0]
    p2r=point[0] + r*math.cos(angle)
    p3r=point[0] + r*math.cos(angle) + c*math.sin(angle)
    p4r=point[0] + c*math.sin(angle)
    
    p1c=point[1]
    p2c=point[1] - r*math.sin(angle)
    p3c=point[1] + c*math.cos(angle) - r*math.sin(angle)
    p4c=point[1] + c*math.cos(angle)
    # 判断旋转后是否在边框内
    if p1r<0 or p2r<0 or p3r<0 or p4r<0 or p1c<0 or p2c<0 or p3c<0 or p4c<0 :
        raise ArithmeticError('cannot fit shape to image')
    if p1r>image[0] or p2r>image[0] or p3r>image[0] or p4r>image[0] or p1c>image[1] or p2c>image[1] or p3c>image[1] or p4c>image[1]:
        raise ArithmeticError('cannot fit shape to image')
            
    rectangle = draw_polygon([
        p1r,
        p2r,
        p3r,
        p4r,
    ], [
        p1c,
        p2c,
        p3c,
        p4c,
    ])
    label = ('rectangle', ((point[0], point[0] + r*math.cos(angle) + c*math.sin(angle)), (point[1] - r*math.sin(angle), point[1] + c*math.cos(angle))))

    return rectangle, label


def _generate_parallelogram_mask(point, image, rshape, random,angle=0):
    available_width = min(image[1] - point[1], rshape[1])
    if available_width < rshape[0]:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image[0] - point[0], rshape[1])
    if available_height < rshape[0]:
        raise ArithmeticError('cannot fit shape to image')
        
    # Pick random widths and heights.
    r = random.randint(rshape[0], available_height + 1)
    c = random.randint(rshape[0], available_width + 1)
    d = random.randint(   1    , available_width + 1)
    
    p1r=point[0]
    p2r=point[0] + r*math.cos(angle) - d*math.sin(angle)
    p3r=point[0] + r*math.cos(angle) + c*math.sin(angle) - d*math.sin(angle)
    p4r=point[0] + c*math.sin(angle)
    
    p1c=point[1]
    p2c=point[1] - r*math.sin(angle) - d*math.cos(angle)
    p3c=point[1] + c*math.cos(angle) - r*math.sin(angle) - d*math.cos(angle)
    p4c=point[1] + c*math.cos(angle)
    # 判断旋转后是否在边框内
    if p1r<0 or p2r<0 or p3r<0 or p4r<0 or p1c<0 or p2c<0 or p3c<0 or p4c<0 :
        raise ArithmeticError('cannot fit shape to image')
    if p1r>image[0] or p2r>image[0] or p3r>image[0] or p4r>image[0] or p1c>image[1] or p2c>image[1] or p3c>image[1] or p4c>image[1]:
        raise ArithmeticError('cannot fit shape to image')
            
    parallelogram = draw_polygon([
        p1r,
        p2r,
        p3r,
        p4r,
    ], [
        p1c,
        p2c,
        p3c,
        p4c,
    ])
    label = ('parallelogram', ((point[0], point[0] + r*math.cos(angle) + c*math.sin(angle)- d*math.sin(angle)), 
                               (point[1] - r*math.sin(angle)- d*math.cos(angle), point[1] + c*math.cos(angle))))

    return parallelogram, label

def _generate_trapezium_mask(point, image, rshape, random,angle=0):
    available_width = min(image[1] - point[1], rshape[1])
    if available_width < rshape[0]:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image[0] - point[0], rshape[1])
    if available_height < rshape[0]:
        raise ArithmeticError('cannot fit shape to image')
        
    # Pick random widths and heights.
    r = random.randint(rshape[0], available_height + 1)
    c = random.randint(rshape[0], available_width + 1)
    d1= random.randint(   -r   , available_width + 1)
    d2= random.randint(   -r   , available_width + 1)
    
    p1r=point[0]
    p2r=point[0] + r*math.cos(angle) - d1*math.sin(angle)
    p3r=point[0] + r*math.cos(angle) + c*math.sin(angle) + d2*math.sin(angle)
    p4r=point[0] + c*math.sin(angle)
    
    p1c=point[1]
    p2c=point[1] - r*math.sin(angle) - d1*math.cos(angle)
    p3c=point[1] + c*math.cos(angle) - r*math.sin(angle) + d2*math.cos(angle)
    p4c=point[1] + c*math.cos(angle)
    # 判断旋转后是否在边框内
    if p1r<0 or p2r<0 or p3r<0 or p4r<0 or p1c<0 or p2c<0 or p3c<0 or p4c<0 :
        raise ArithmeticError('cannot fit shape to image')
    if p1r>image[0] or p2r>image[0] or p3r>image[0] or p4r>image[0] or p1c>image[1] or p2c>image[1] or p3c>image[1] or p4c>image[1]:
        raise ArithmeticError('cannot fit shape to image')
    #判断梯形下边两点不接触
    if (d1+d2)<=-r:
        raise ArithmeticError('cannot fit shape to image')
        
    trapezium = draw_polygon([
        p1r,
        p2r,
        p3r,
        p4r,
    ], [
        p1c,
        p2c,
        p3c,
        p4c,
    ])
    label = ('trapezium',((point[0], point[0] + r*math.cos(angle) + c*math.sin(angle)+ d2*math.sin(angle)), 
                          (point[1] - r*math.sin(angle)- d1*math.cos(angle), point[1] + c*math.cos(angle))))

    return trapezium, label

def _generate_ellipse_mask(point, image, rshape, random, angle=0):
    if rshape[0] == 1 or rshape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = rshape[0] / 2.0
    max_radius = rshape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    radius = random.randint(min_radius, available_radius + 1)
    import random as rd
    radius1=radius*(rd.random()/2 + 0.5)
    radius2=radius*(rd.random()/2 + 0.5)
    if radius1>radius2:
        r_radius=radius1
        c_radius=radius2
    else:
        r_radius=radius2
        c_radius=radius1
    ellipse = draw_ellipse(point[0], point[1], r_radius, c_radius, rotation=angle)
    label = ('ellipse', ((point[0] - radius + 1, point[0] + radius),
                        (point[1] - radius + 1, point[1] + radius)))

    return ellipse, label

def _generate_circle_mask(point, image, rshape, random,angle=0):

    if rshape[0] == 1 or rshape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = rshape[0] / 2.0
    max_radius = rshape[1] / 2.0
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom, max_radius)
    if available_radius < min_radius:
        raise ArithmeticError('cannot fit shape to image')
    radius = random.randint(min_radius, available_radius + 1)
    circle = draw_circle(point[0], point[1], radius)
    label = ('circle', ((point[0] - radius + 1, point[0] + radius),
                        (point[1] - radius + 1, point[1] + radius)))

    return circle, label


def _generate_triangle_mask(point, image, rshape, random,angle=0):
    if rshape[0] == 1 or rshape[1] == 1:
        raise ValueError('dimension must be > 1 for triangles')
    available_side = min(image[1] - point[1], point[0] + 1, rshape[1])
    if available_side < rshape[0]:
        raise ArithmeticError('cannot fit shape to image')

    side0 = random.randint(rshape[0], available_side + 1)
    side = random.randint(rshape[0], available_side + 1)
    angle2 = math.pi/2-angle-math.acos(math.sqrt((2*side**2-side0**2)/(2*side**2)))
    #angle2=math.pi/2-angle-math.pi/3
    triangle_height = int(np.ceil(np.sqrt(3 / 4.0) * side))
    

    p1r=point[0]
    p2r=point[0] - side*math.sin(angle)
    p3r=point[0] - side*math.cos(angle2)
    
    p1c=point[1]
    p2c=point[1] + side*math.cos(angle)
    p3c=point[1] + side*math.sin(angle2)
    
    
    if p1r<0 or p2r<0 or p3r<0 or p1c<0 or p2c<0 or p3c<0 :
        raise ArithmeticError('cannot fit shape to image')
    if p1r>image[0] or p2r>image[0] or p3r>image[0] or p1c>image[1] or p2c>image[1] or p3c>image[1] :
        raise ArithmeticError('cannot fit shape to image')
    
    triangle = draw_polygon([
        p1r,
        p2r,
        p3r,
    ], [
        p1c,
        p2c,
        p3c,
    ])
    label = ('triangle', ((point[0] - triangle_height, point[0]),
                          (point[1], point[1] + side)))

    return triangle, label


# Allows lookup by key as well as random selection.
SHAPE_GENERATORS = dict(
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask,
    triangle=_generate_triangle_mask,
    ellipse=_generate_ellipse_mask,
    parallelogram=_generate_parallelogram_mask,
    trapezium=_generate_trapezium_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())

#%%将属性转化为图片
import matplotlib.pyplot as plt

shape,angle,location,color,size,min_size,max_size,=random_attributes(2,20,80,2,2,2,2)

image_shape=(224,224)
rshape=(min_size,max_size)

num_channels=3
shape_generator = SHAPE_GENERATORS[shape]
num_trials=10000


image_shape = (image_shape[0], image_shape[1], num_channels)
image = np.full(image_shape, 255, dtype=np.uint8)
filled = np.zeros(image_shape, dtype=bool)
for _ in range(num_trials):
# Pick start coordinates.
    column = random.randint(0,image_shape[1])
    row = random.randint(0,image_shape[0])
    point = (row, column)
    try:
        indices, label = shape_generator(point, image_shape, rshape,
                                                 random,angle)
    except ArithmeticError:
                # Couldn't fit the shape, skip it.
        continue



file = open('zm.txt','w') 
fig=plt.figure(figsize=(224,224))
fig.set_size_inches(0.24/3,0.24/3) 
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)

fig.savefig(format='png', transparent=True, dpi=2800, pad_inches = 0)
