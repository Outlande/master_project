import os
import sys
sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import cv2
import time
from hfnet import HFNet
import argparse
import glob
import numpy as np
from tqdm import tqdm

def main(args):
    print("Load hfnet Start")
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
    hfnet = HFNet(args.model_path, outputs)
    print("Load hfnet Finished")

    print("start processing folder: ", args.image_folder)
    imgs = glob.glob(args.image_folder + "/*.png")
    print(" image nums: ", len(imgs))
    for i in tqdm(range(len(imgs))):
        path = imgs[i]
        image_name = os.path.basename(path)[:-4]
        image = cv2.imread(path)[:, :, [2,1,0]]
        out = hfnet.inference(image)
        np.savetxt(args.out_folder+"/"+image_name+"_gdes.txt", out["global_descriptor"])
        np.savetxt(args.out_folder+"/"+image_name+"_ldes.txt", out["local_descriptors"])
        np.savetxt(args.out_folder+"/"+image_name+"_kpts.txt", out["keypoints"])

        image_show = cv2.imread(path)
        for idx in range(out["local_descriptors"].shape[0]):
            cv2.circle(image_show,center=out["keypoints"][idx],radius=2,color=(0, 255, 0),thickness=-1)
        cv2.imwrite(args.out_folder+"/"+image_name+"_show.png", image_show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/home/levi/master_project/hfnet/hfnet_model")
    parser.add_argument('--image_folder', type=str, default = None)
    parser.add_argument('--out_folder', type=str, default = None)
    args = parser.parse_args()
    main(args)

