# -*- coding: utf-8 -*-
"""
@author: Krieg0-0 and Blink0621

"""

#%% 主函数 包含三角形、圆、椭圆、梯形、平行四边形、矩形


import numpy as np
import math

from skimage.draw import polygon as draw_polygon, circle as draw_circle,ellipse as draw_ellipse
from skimage._shared.utils import warn


def _generate_rectangle_mask(point, image, size,p, random,OorA,lean,angle=0):
    
    r = int(math.sqrt(size / p))
    c = int(math.sqrt(size * p))
    
    
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
    label = ('rectangle', ((p1r,p3r),(p1c,p3c)))

    return rectangle, label

#%%未想好平行四边形与p(长宽比)与size(面积)求两边长， 梯形也是如此
def _generate_parallelogram_mask(point, image, size,p, random,OorA,lean,angle=0):
    r = int(math.sqrt(size / p))
    c = int(math.sqrt(size * p))
    
    
    d = int(OorA * c)
    
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
    label = ('parallelogram', ((p1r,p3r),(p1c,p3c)))

    return parallelogram, label

def _generate_trapezium_mask(point, image, size,p, random,OorA,lean,angle=0):
    
    if OorA < 0.5 :
        B=int(math.sqrt(size*2/p/(lean+1)))
        H=int(B*p)
        d=int(B*(1-lean)/2)
    else:
        B=int(math.sqrt(size*2*p/(lean+1)))
        H=int(B/p)
        d=int(B*(1-lean)/2)
    
    if image[1] - point[1] < H:
        raise ArithmeticError('cannot fit shape to image')
    if image[0] - point[0] < B:
        raise ArithmeticError('cannot fit shape to image')
    
    p1r=point[0] 
    p2r=point[0] + B*math.cos(angle)
    p3r=point[0] + B*math.cos(angle) - H*math.sin(angle) - d*math.cos(angle)
    p4r=point[0] - H*math.sin(angle) + d*math.cos(angle)
    
    p1c=point[1] 
    p2c=point[1] + B*math.sin(angle)
    p3c=point[1] + B*math.sin(angle) + H*math.cos(angle) - d*math.sin(angle)
    p4c=point[1] + H*math.cos(angle) + d*math.sin(angle)
    # 判断旋转后是否在边框内
    if p1r<0 or p2r<0 or p3r<0 or p4r<0 or p1c<0 or p2c<0 or p3c<0 or p4c<0 :
        raise ArithmeticError('cannot fit shape to image')
    if p1r>image[0] or p2r>image[0] or p3r>image[0] or p4r>image[0] or p1c>image[1] or p2c>image[1] or p3c>image[1] or p4c>image[1]:
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
    label = ('trapezium',((p1r,p3r + d*math.cos(angle)), 
                          (p1c,p3c + d*math.sin(angle))))

    return trapezium, label
#%%
def _generate_ellipse_mask(point, image, size,p, random, OorA,lean,angle=0):
    if size == 1 :
        raise ValueError('size must be > 1 for ellipses')
        
        
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius=min(left,right,top,bottom)
    
    r_radius = int(math.sqrt((size / p) / math.pi))
    c_radius = int(math.sqrt((size * p) / math.pi))
    
    if r_radius > available_radius :
        raise ArithmeticError('cannot fit shape to image')

    ellipse = draw_ellipse(point[0], point[1], r_radius, c_radius, rotation=angle)
    
    #不是很懂为啥有+1
    label = ('ellipse', ((point[0] - r_radius + 1, point[0] + r_radius),
                        (point[1] - r_radius + 1, point[1] + r_radius)))
    #
    return ellipse, label
#%%
def _generate_circle_mask(point, image, size,p, random,OorA,lean,angle):

    if size == 1 :
        raise ValueError('size must be > 1 for circles')
        
        
    left = point[1]
    right = image[1] - point[1]
    top = point[0]
    bottom = image[0] - point[0]
    available_radius = min(left, right, top, bottom)
    
    radius = int(math.sqrt(size / math.pi))
    
    if radius > available_radius :
        raise ArithmeticError('cannot fit shape to image')

    circle = draw_circle(point[0], point[1], radius)
    label = ('circle', ((point[0] - radius + 1, point[0] + radius),
                        (point[1] - radius + 1, point[1] + radius)))

    return circle, label

#%%
def _generate_triangle_mask(point, image, size,p, random,OorA,lean,angle=0):
    if size == 1 :
        raise ValueError('dimension must be > 1 for triangles')
    r = int(math.sqrt(2*size / p))
    c = int(math.sqrt(2*size * p))
    
    if OorA<0.5:
        B = r
        H = c
    else:
        B = c
        H = r
        
    p1r=point[0]
    p2r=point[0] + B * math.cos(angle)
    p3r=point[0] + B * math.cos(angle) / 2 - H * math.sin(angle)
    
    p1c=point[1]
    p2c=point[1] + B * math.sin(angle)
    p3c=point[1] + H * math.cos(angle) + B * math.sin(angle) / 2
    
    
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
    label = ('triangle', ((p1r,p3r + B/2*math.sin(angle))),
                          (p1c,p3c))

    return triangle, label


#%% Allows lookup by key as well as random selection.
SHAPE_GENERATORS = dict(
    rectangle=_generate_rectangle_mask,
    circle=_generate_circle_mask,
    triangle=_generate_triangle_mask,
    ellipse=_generate_ellipse_mask,
    parallelogram=_generate_parallelogram_mask,
    trapezium=_generate_trapezium_mask)
SHAPE_CHOICES = list(SHAPE_GENERATORS.values())



def _generate_random_colors(num_colors, num_channels, intensity_range, random):

    if num_channels == 1:
        intensity_range = (intensity_range, )
    elif len(intensity_range) == 1:
        intensity_range = intensity_range * num_channels
    colors = [random.randint(r[0], r[1]+1, size=num_colors)
              for r in intensity_range]
    return np.transpose(colors)

def random_shapes(size,
                  p,
                  image_shape,
                  max_shapes,
                  min_shapes=1,
                  multichannel=True,
                  num_channels=3,
                  shape=None,
                  intensity_range=None,
                  allow_overlap=False,
                  num_trials=10000,
                  random_seed=None):


    if not multichannel:
        num_channels = 1

    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254), )
    else:
        tmp = (intensity_range, ) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    random = np.random.RandomState(random_seed)
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.full(image_shape, 255, dtype=np.uint8)
    filled = np.zeros(image_shape, dtype=bool)
    labels = []

    num_shapes = random.randint(min_shapes, max_shapes + 1)
    colors = _generate_random_colors(num_shapes, num_channels,
                                     intensity_range, random)
    angle = np.random.uniform(-math.pi,math.pi)
    OorA=np.random.random()
    lean=np.random.random()
    
    if shape is None:
        shape_generator = random.choice(SHAPE_CHOICES)
    else:
        shape_generator = SHAPE_GENERATORS[shape]
    ####为了使生成的相同的几个图形产生相同的颜色而进行的值传递
    A = range(num_shapes)
    ####
    for shape_idx in A:
        for _ in range(num_trials):
            # Pick start coordinates.
            column = random.randint(image_shape[1])
            row = random.randint(image_shape[0])
            point = (row, column)
            try:
                indices, label = shape_generator(point, image_shape,size,
                                                 p,random,OorA,lean,angle)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                image[indices] = colors[A[0]]
                filled[indices] = True
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    if not multichannel:
        image = np.squeeze(image, axis=2)
    return image, labels, angle, colors, shape, size


#%% 生成图片
import matplotlib.pyplot as plt
import random

file = open('info.txt','w') #创建一个txt文档，向其中写入信息
n=0
while n < 2:
    
    try:
        size=random.randint(100,5000)
        p=random.uniform(0,1)
        fig=plt.figure(figsize=(224,224))
        result = random_shapes(size,
                               p,
                               image_shape=(224,224),
                               max_shapes=1,
                               intensity_range=((0, 255),),)
        image, labels, angle, colors, shape,size= result
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        
    except ValueError:
        plt.close('all')
        pass
        
    else:
        n=n+1
        fig.set_size_inches(0.24/3,0.24/3) 
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        filename='%d.png'%n
        fig.savefig(fname=filename,format='png', transparent=True, dpi=2800, pad_inches = 0)
        file.write('picture:{}  Shape: {}      Angle:{:.2f}    Location:{}     Color:{}    Size:{}\n'.format(n,labels[0][0],angle,labels[0][1],colors[0],size))
        plt.close('all')
file.close()
