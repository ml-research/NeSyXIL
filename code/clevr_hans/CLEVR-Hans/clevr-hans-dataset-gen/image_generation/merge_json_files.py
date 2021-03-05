import json
import glob
import argparse
import random
import numpy as np
import os
from pycocotools import mask

parser = argparse.ArgumentParser(description='PyTorch Covid19 Training')
parser.add_argument('--json_dir', default="/home/ml-wstammer/WS/datasets/Clevr-Hans/mini/test/",
					type=str, help='dir to json files to be merged')
parser.add_argument('--convert_to_rle', action='store_true',
					help='is it necessary to convert compressed str values to uncompressed rles?')
args = parser.parse_args()


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


def str_to_biimg(imgstr):
	img = []

	cur = 0
	for num in imgstr.strip().split(','):
		num = int(num)
		img += [cur] * num
		cur = 1 - cur
	return np.array(img)


def main():
	# json_dir = '/Users/ml-wstammer/Documents/PhD/repositories/XIL_VQA/clevr-dataset-gen/output/mini/conf/test/'
	json_list = glob.glob('{}CLEVR_HANS_*.json'.format(args.json_dir))

	for i, json_fp in enumerate(json_list):
		with open(json_fp) as json_file:
			d = json.load(json_file)

		if args.convert_to_rle:
			print("Converting json file rles from {}".format(json_fp))
			# first convert mask annotations in dictionary to compressed rle
			for img_id, scene in enumerate(d['scenes']):
				for obj_id, obj in enumerate(scene['objects']):
					rle = obj['mask']

					mask_pixel_num = rle.get('size')[0] * rle.get('size')[1]
					gt_mask = np.array([0] * mask_pixel_num)
					obj_mask = rle.get('counts')

					gt_mask |= str_to_biimg(obj_mask)
					gt_mask = gt_mask.reshape((320, 480))

					rle = mask.frPyObjects(encodeMask(gt_mask), rle.get('size')[0], rle.get('size')[1])
					rle['counts'] = rle.get('counts').decode('utf-8')

					d['scenes'][img_id]['objects'][obj_id]['mask'] = rle

			print("Converted json file rles from {}".format(json_fp))

		# append all scene lists together
		if i == 0:
			scenes_dict = d.copy()
			scenes_dict['info'] = "CLEVR-Hans dataset annotations with object mask in coco rle format"
		else:
			scenes_dict['scenes'] += d['scenes']

	# shuffle scenes
	random.shuffle(scenes_dict['scenes'])

	save_fp = '{}CLEVR_HANS_scenes_{}.json'.format(args.json_dir, args.json_dir.split(os.sep)[-2])
	with open(save_fp, 'w') as outfile:
		json.dump(scenes_dict, outfile)
	print("JSON files under {} were merged into {}".format(args.json_dir, save_fp))


if __name__ == "__main__":
	main()



