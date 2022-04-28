import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


################################################################################
def compute_delf(imgs):
    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    D = []
    for img in imgs:
        D.append(compute_local_descriptor(img, delf))

    return D


################################################################################
def compute_local_descriptor(img, delf):
    im_height = img.shape[0]
    im_width = img.shape[1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    out = delf(image=img,
               score_threshold=tf.constant(0.0),
               image_scales=tf.constant([1.0]),
               max_feature_num=tf.constant(200))

    return {'descriptors': np.array(out['features']),
            'descriptors_pca': np.array(out['descriptors']),
            'scores': np.array(out['scores']),
            'keypoints': np.array(out['locations']),
            'scales': 1./np.array(out['scales']),
            'imheight': np.array(im_height),
            'imwidth': np.array(im_width)
            }
