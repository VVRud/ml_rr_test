import argparse
import os
import numpy as np
from PIL import Image
from PIL.ImageStat import Stat
from PIL import ImageChops as ops


def dir_path(path):
    """
    Check if argument is path.
    
    :param path: path to check
    :returns: path back if it is valid
    :raises ArgumentTypeError: raises if path is not valid
    """
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("directory: '{}' is not valid.".format(path))

def average_hash(im, size):
    """
    Calculates average hash as described in
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    :param im: PIL Image object
    :param size: size of the side for resized image
    :returns: binary numpy ndarray of shape (size*size, )
    """
    d = np.array(im.resize((size, size)).convert('L'))
    
    return (d.ravel() > d.mean()).astype(np.int32)

def dv_hash(im, size):
    """
    Calculates gradient hash as described in
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

    :param im: PIL Image object
    :param size: size of the side for resized image
    :returns: binary numpy ndarray of shape (size*size, )
    """
    d = np.array(im.resize((size, size + 1)).convert('L'))
    
    return (d[1:, :] > d[:-1, :]).ravel().astype(np.int32)

def hamming_distance(h1, h2):
    """
    Calculates Hamming distance between 2 arrays.

    :param h1: first array
    :param h2: second array
    :returns: Hamming distance
    """
    return np.sum(h1 != h2)

def stats(a):
    """
    Calculate stats as described in
    https://habr.com/ru/post/122372/

    :param a: numpy ndarray of shape (2, 2)
    :returns: numpy ndarray of shape (4, ) with calculated stats
    """
    i0 = (a[0, 0] + a[0, 1] + a[1, 0] + a[1, 1])/4
    i1 = 128 + (a[0, 0] - a[0, 1] + a[1, 0] - a[1, 1])/4
    i2 = 128 + (a[0, 0] + a[0, 1] - a[1, 0] - a[1, 1])/4
    i3 = 128 + (a[0, 0] - a[0, 1] - a[1, 0] + a[1, 1])/4
    
    return np.stack([i0, i1, i2, i3])

def rms(a, b):
    """
    Calculate by-pixel root mean squared error

    :param a: numpy array of first image
    :param b: numpy array of second image
    :returns: root mean squared error
    """
    return np.sqrt(np.power(a - b, 2).mean())

def calc_stats(im1, im2):
    """
    Calculate statistics to use in linear regression.

    :param im1: PIL Image object
    :param im2: PIL Image object
    :returns: numpy ndarray of shape (12, )
    """
    ph = []
    or1 = im1.convert('L')
    or2 = im2.convert('L')
    ds1 = or1.resize((1024, 1024))
    ds2 = or2.resize((1024, 1024))

    ph.append(np.abs(stats(np.array(or1.resize((2, 2)), dtype=np.float32)) - 
                stats(np.array(or2.resize((2, 2)), dtype=np.float32))))
    ph.append([rms(np.array(ds1), np.array(ds2))])
    ph.append([np.abs(Stat(or1).mean[0] - Stat(or2).mean[0])])
    ph.append([np.abs(Stat(or1).stddev[0] - Stat(or2).stddev[0])])

    ph.append(Stat(ops.lighter(ds1, ds2)).mean)
    ph.append(Stat(ops.darker(ds1, ds2)).mean)
    ph.append(Stat(ops.lighter(ds1, ds2)).stddev)
    ph.append(Stat(ops.darker(ds1, ds2)).stddev)

    ph.append([np.abs(or1.size[0] / or1.size[1] - or2.size[0] / or2.size[1])])
    return np.hstack(ph)

def predict(x, w):
    """
    Makes prediction for logistic regression.

    :param x: numpy ndarray of features and examples
    :param w: numpy ndarray matrix of weights
    :returns: vector of predictions
    """
    p = x.dot(w)
    return 1. / (1. + np.exp(-p))

def classify_rulebased(im1, im2):
    """
    Classify 2 images using rule-based method.

    :param im1: PIL Image object
    :param im2: PIL Image object
    :returns: False if images are different, True otherwise
    """
    or1 = im1.convert('L')
    or2 = im2.convert('L')
    ds1 = or1.resize((1024, 1024))
    ds2 = or2.resize((1024, 1024))

    err = rms(np.array(ds1), np.array(ds2))
    size_diff = np.abs(or1.size[0] / or1.size[1] - or2.size[0] / or2.size[1])
    d_std = Stat(ops.darker(ds1, ds2)).stddev[0]

    if err < 9.8 and (err == 0 or size_diff < 0.5 or d_std > 28):
        return True
    else:
        return False

def classify_regression(im1, im2, thr=0.6):
    """
    Classify 2 images using logistic regression.

    :param im1: PIL Image object
    :param im2: PIL Image object
    :param thr: threshol to use for prediction
    :returns: False if images are different, True otherwise
    """
    w = np.array([0.0007282, -0.01936689, -0.01438374, -0.039445,
                -0.05667355, -0.0250665, -0.13507414, -0.07393462,
                -0.15777153, 0.13417615, 0.1085812, 0.12869122, -0.00268428],
                dtype=np.float32)
    X = np.hstack([[1], calc_stats(im1, im2)])
    return predict(X, w) > thr

def classify_hashbased(im1, im2):
    """
    Classify 2 images using hash-based method.

    :param im1: PIL Image object
    :param im2: PIL Image object
    :returns: False if images are different, True otherwise
    """
    avg = hamming_distance(average_hash(im1, 32), average_hash(im2, 32))
    dv = hamming_distance(dv_hash(im1, 32), dv_hash(im2, 32))
    if avg < 300 and dv < 475:
        return True
    else:
        return False

def classify_ensemble(im1, im2):
    """
    Classify 2 images using ensembling.

    :param im1: PIL Image object
    :param im2: PIL Image object
    :returns: False if images are different, True otherwise
    """
    return sum([
        classify_rulebased(im1, im2),
        classify_regression(im1, im2),
        classify_hashbased(im1, im2)
    ]) >= 2

def print_names(im1, im2):
    """
    Print 2 images names.

    :param im1: PIL Image object
    :param im2: PIL Image object
    :returns: None
    """
    n1 = im1.filename.split(os.sep)[-1]
    n2 = im2.filename.split(os.sep)[-1]
    print(n1, n2)

def main(path, method):
    """
    Reads images and perform classification on each pair.

    :param path: path to folder where images are located
    :param method: method to do classification with
    """
    im_list = [Image.open(os.path.join(path, img)) for img in os.listdir(path) if not os.path.isdir(img)]
    classify = {
        'rule': classify_rulebased,
        'regression': classify_regression,
        'hash': classify_hashbased,
        'ensemble': classify_ensemble
    }

    for i in range(len(im_list)):
        for j in range(i + 1, len(im_list)):
            try:
                if classify[method](im_list[i], im_list[j]):
                    print_names(im_list[i], im_list[j])
            except KeyError as e:
                raise argparse.ArgumentTypeError("method: '{}' is not valid.".format(method))

if __name__ == "__main__":
    """
    Create argument parser, parse args and run main method.
    """
    parser = argparse.ArgumentParser(description='First test task on images similarity.')
    parser.add_argument('--path', metavar='PATH', required=True, type=dir_path, help='folder with images')
    parser.add_argument('--method', metavar='METH', default='ensemble', type=str,
                        help="""rule - rule-based method;
                        regression - logistic regression;
                        hash - hash-based method;
                        ensemble - ensemble-based method""")
    
    args = parser.parse_args()
    main(args.path, args.method)