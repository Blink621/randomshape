# -*- coding: utf-8 -*-
"""
@author: Blink0621

"""

#%% 主函数 包含三角形、圆、椭圆、梯形、平行四边形、矩形


import numpy as np
import math

from skimage.draw import polygon as draw_polygon, circle as draw_circle,ellipse as draw_ellipse
from skimage._shared.utils import warn


def _generate_rectangle_mask(point, image, shape, random,angle=0):
    available_width = min(image[1] - point[1], shape[1])
    if available_width < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image[0] - point[0], shape[1])
    if available_height < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
        
    # Pick random widths and heights.
    r = random.randint(shape[0], available_height + 1)
    c = random.randint(shape[0], available_width + 1)
    
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


def _generate_parallelogram_mask(point, image, shape, random,angle=0):
    available_width = min(image[1] - point[1], shape[1])
    if available_width < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image[0] - point[0], shape[1])
    if available_height < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
        
    # Pick random widths and heights.
    r = random.randint(shape[0], available_height + 1)
    c = random.randint(shape[0], available_width + 1)
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

def _generate_trapezium_mask(point, image, shape, random,angle=0):
    available_width = min(image[1] - point[1], shape[1])
    if available_width < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
    available_height = min(image[0] - point[0], shape[1])
    if available_height < shape[0]:
        raise ArithmeticError('cannot fit shape to image')
        
    # Pick random widths and heights.
    r = random.randint(shape[0], available_height + 1)
    c = random.randint(shape[0], available_width + 1)
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

def _generate_ellipse_mask(point, image, shape, random, angle=0):
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
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

def _generate_circle_mask(point, image, shape, random):

    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('size must be > 1 for circles')
    min_radius = shape[0] / 2.0
    max_radius = shape[1] / 2.0
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


def _generate_triangle_mask(point, image, shape, random,angle=0):
    if shape[0] == 1 or shape[1] == 1:
        raise ValueError('dimension must be > 1 for triangles')
    available_side = min(image[1] - point[1], point[0] + 1, shape[1])
    if available_side < shape[0]:
        raise ArithmeticError('cannot fit shape to image')

    side0 = random.randint(shape[0], available_side + 1)
    side = random.randint(shape[0], available_side + 1)
    angle2=math.pi/2-angle-math.acos(math.sqrt((2*side**2-side0**2)/(2*side**2)))
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



def _generate_random_colors(num_colors, num_channels, intensity_range, random):

    if num_channels == 1:
        intensity_range = (intensity_range, )
    elif len(intensity_range) == 1:
        intensity_range = intensity_range * num_channels
    colors = [random.randint(r[0], r[1]+1, size=num_colors)
              for r in intensity_range]
    return np.transpose(colors)

def random_shapes(image_shape,
                  max_shapes,
                  min_shapes=1,
                  min_size=2,
                  max_size=None,
                  multichannel=True,
                  num_channels=3,
                  shape=None,
                  intensity_range=None,
                  allow_overlap=False,
                  num_trials=100000,
                  random_seed=None):
    if min_size > image_shape[0] or min_size > image_shape[1]:
        raise ValueError('Minimum dimension must be less than ncols and nrows')
    max_size = max_size or max(image_shape[0], image_shape[1])

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
    user_shape = shape
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.full(image_shape, 255, dtype=np.uint8)
    filled = np.zeros(image_shape, dtype=bool)
    labels = []

    num_shapes = random.randint(min_shapes, max_shapes + 1)
    colors = _generate_random_colors(num_shapes, num_channels,
                                     intensity_range, random)
    if user_shape is None:
        shape_generator = random.choice(SHAPE_CHOICES)
    else:
        shape_generator = SHAPE_GENERATORS[user_shape]
        
    shape = (min_size, max_size)
    angle = random.uniform(-math.pi,math.pi)
    
    for shape_idx in range(num_shapes):
        for _ in range(num_trials):
            # Pick start coordinates.
            column = random.randint(image_shape[1])
            row = random.randint(image_shape[0])
            point = (row, column)
            try:
                indices, label = shape_generator(point, image_shape, shape,
                                                 random,angle)
            except ArithmeticError:
                # Couldn't fit the shape, skip it.
                continue
            # Check if there is an overlap where the mask is nonzero.
            if allow_overlap or not filled[indices].any():
                image[indices] = colors[shape_idx]
                filled[indices] = True
                labels.append(label)
                break
        else:
            warn('Could not fit any shapes to image, '
                 'consider reducing the minimum dimension')

    if not multichannel:
        image = np.squeeze(image, axis=2)
    return image, labels
      
#%% 生成图片
import matplotlib.pyplot as plt


n=0
while n < 2:
    
    try:
        fig=plt.figure(figsize=(224,224))
        result = random_shapes((224,224),
                               min_shapes=1, 
                               max_shapes=1,
                               intensity_range=((0, 255),),
                               min_size=20,
                               max_size=80)
        image, labels = result
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        
    except BaseException:
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
        plt.close('all')
