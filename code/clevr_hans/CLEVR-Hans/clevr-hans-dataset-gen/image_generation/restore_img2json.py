import cv2
import sys
import os
import json
import numpy as np
from pycocotools import mask as maskUtils

def encodeMask(M):
  """
  Encode binary mask M using run-length encoding.
  :param   M (bool 2D array)  : binary mask to encode
  :return: R (object RLE)     : run-length encoding of binary mask
  """
  [h, w] = M.shape
  M = M.flatten(order='F')
  N = len(M)
  counts_list = []
  pos = 0
  # counts
  counts_list.append(1)
  diffs = np.logical_xor(M[0:N - 1], M[1:N])
  for diff in diffs:
    if diff:
      pos += 1
      counts_list.append(1)
    else:
      counts_list[pos] += 1
  # if array starts from 1. start with 0 counts for 0
  if M[0] == 1:
    counts_list = [0] + counts_list
  return {'size': [h, w],
          'counts': counts_list,
          }


def main(index, idx):
  obj_mask = {}
  for _i in range(index):
    path = './tmp_graph_output/indexob{}_{}/Image0001.png'.format(_i,idx)
    print('read path... {}'.format(path))

    # compute uncompressed rle from binary mask
    rle = encodeMask(cv2.imread(path)[:,:,0])
    # compute compressed rle
    rle = maskUtils.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    # decode the bytes object to str
    rle['counts'] = rle.get('counts').decode('utf-8')

    obj_mask[_i+1] = rle
    path = './tmp_graph_output/indexob{}_{}/'.format(_i, idx)
    os.system('rm -r {}'.format(path))

  json.dump(obj_mask, open('/tmp/obj_mask_{}.json'.format(idx), 'w'))


main(int(sys.argv[1]), int(sys.argv[2]))
