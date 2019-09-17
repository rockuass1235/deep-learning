import sys
import os
from mxnet import image, nd
import dlib
import time
import _pickle as pk




def read_img(path):
    if path[-1] != '/':
        path += '/'

    li = []
    if os.path.isdir(path):
        files = os.listdir(path)

        for file in files:
            li += read_img(path + file)

    else:
        li.append(path[:-1])
    return li


MAX_gt = 20
def get_ground_truth(img):
    h, w, _ = img.shape
    detector = dlib.cnn_face_detection_model_v1('mod/mmod_human_face_detector.dat')
    gts = nd.ones(shape=(MAX_gt, 5)) * -1
    dets = detector(img, 1)

    if len(dets) <= 0:
        return (False, gts)

    for i, det in enumerate(dets):
        if i >= 20:
            break
        x1, y1, x2, y2 = det.rect.left() / w, det.rect.top() / h, det.rect.right() / w, det.rect.bottom() / h
        gts[i] = nd.array([0, x1, y1, x2, y2])

    return (True, gts)


if __name__ == '__main__':

   args = sys.argv[1:]
   data_path = args[0]
   file_name = args[1]
   threshold = float(args[2])



   lst = read_img(data_path)
   lst = [path[len(data_path):] for path in lst]
   X = []
   Y = []
   n = len(lst)

   start = time.time()
   for i in range(n):

       img = image.imread(data_path + lst[i]).asnumpy()

       flg, gts = get_ground_truth(img)

       if flg:
           X.append(lst[i])
           Y.append(gts.expand_dims(axis=0))

       print('%d / %d' % (i + 1, n))

   Y = nd.concat(*Y, dim = 0)



   print('start writting  files.......')
   n = int(threshold * len(X))



   with open(file_name + '_train.lst', 'wb') as f:
       pk.dump(X[:n], f)  # data type: list
   with open(file_name + '_train.rec', 'wb') as f:
       pk.dump(Y[:n].asnumpy(), f)  # data type: numpy

   with open(file_name + '_test.lst', 'wb') as f:
       pk.dump(X[n:], f)  # data type: list
   with open(file_name + '_test.rec', 'wb') as f:
       pk.dump(Y[n:].asnumpy(), f)  # data type: numpy


   print('success !!!!')
   print('total time: %f' % (time.time() - start))


