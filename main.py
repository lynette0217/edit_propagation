import cv2
import os
from os import path
import numpy as np
import argparse
import torch
from interactive import *
from solver import *
from scipy.sparse import dia_matrix


def make_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--alg", type=str, default="AppProp", help="the algorithm you want to choose for edit propogation.",
       choices=["AppProp", "KDtree", "sketch"])

    aa("--img", type=str, default="1",
       help="the path of the image you want to process.")

    aa("--m", type=int, default=-1,
       help="the column of the low rank matrix, U is n*m.")
    aa("--k", type=int, default=200, help="the number of columns for sketch matrix S")
    aa("--operation", type=int, default=1,
       help="0 means increase the brightness. 1 makes the picture bluer. 2 for white")
    aa("--beta", type=float, default=40,
        help="the parameter for the operation, eg.40 for the brightness increasement.")

    # parameters to be tuned
    aa("--weight_edited", type=float, default=5,
       help="0 for non- edited samples and larger values for edited samples")
    aa("--weight_nonedited", type=float, default=0.01,
       help="0 for non- edited samples and larger values for edited samples")

    aa("--sigma_a", type=float, default=50,
        help="500 for low dynamic range images, 0.2 for HDR radiance maps, and 0.2 for materials")
    aa("--sigma_s", type=float, default=100,
        help="we use the value 0.05 for imprecise dense edits to maintain their basic structure, while 10 is used for very sparse ones to propagate as far as possible")

    return parser


if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    print(args)
    defaults = parser.parse_args([])
    path = "./figs/"+args.img + ".png"
    mp = MaskPainter(image_path=path, operation=args.operation)
    mask_path = mp.paint_mask()
    cv2.destroyAllWindows()
    origin_img = cv2.imread(path)
    lab_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2LAB)
    img_rows = origin_img.shape[0]
    img_cols = origin_img.shape[1]
    print("-----Finish the origin image reading process, the image is %d * %d." %
          (img_rows, img_cols))
    mask_img = cv2.imread(mask_path)
    n = img_rows * img_cols
    if args.m == -1 or args.alg == "sketch":
        m = n
    else:
        m = args.m

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    lambda_result, g, W = basic_calculation(
        args.weight_edited, args.weight_nonedited, lab_img, mask_img, args.operation, args.beta)
    # print(lambda_result)
    # print(g)
    # print(W)
    # exit(0)
    output_img = np.zeros((img_rows, img_cols, 3))
    # U = affinity_calculation(
    # origin_img, m, n, args.sigma_a, args.sigma_s)
    U = affinity_calculation(
        lab_img, m, n, args.sigma_a, args.sigma_s)
    # for i in range(3):  # for three channels, calculate each
    #     print("---------BEGIN THE %d CHANNEL CALCULATION---------" % (i+1))
    # sketch_vector = torch.randint(m, [args.k, n]).int()

    # sketch_value = (np.random.normal(
    #                 size=[args.k, n]).astype("float32"))
    # S = np.zeros((m, n))
    # S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(
    #     n).repeat(args.k)] = sketch_value.reshape(-1)
    # U = np.dot(S, U)

    if args.alg == "AppProp":
        e = appProp_lra_calculation(U, g, W, m, n, lambda_result)
    # elif args.alg == "sketch":

    output_lab = e.reshape((img_rows, img_cols, 3))
    output_lab = output_lab.astype('uint8')
    print(output_lab)
    output_img = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)

    print("ALL THE CALCULATION HAS FINISHED!")
    out_path = "./figs/results/"
    cv2.imwrite(out_path+"img=="+args.img+"---w_e="+str(args.weight_edited)+"-w_n="+str(args.weight_nonedited)
                + "-operation="+str(args.operation)+"-beta="+str(args.beta)+"-sigma_a="+str(args.sigma_a)+"-sigma_s="+str(args.sigma_s)+".png", output_img)
