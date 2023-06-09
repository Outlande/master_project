conda activate dsfeat

rm -r /home/levi/master_project/datasets_tum/results
mkdir -p /home/levi/master_project/datasets_tum/results

# iffdetector
rm -r /home/levi/master_project/datasets_tum/results/detect_results/
mkdir -p /home/levi/master_project/datasets_tum/results/detect_results/

cd /home/levi/master_project/iffdetector
./darknet detector folder cfg/tum.data cfg/tum.cfg  weights/final.weights /home/levi/master_project/datasets_tum/rgbd_dataset_freiburg3_walking_halfsphere/rgb/
cd /home/levi/master_projects

# dsfeat
rm -r /home/levi/master_project/datasets_tum/results/dsfeat_results/
mkdir -p /home/levi/master_project/datasets_tum/results/dsfeat_results/

python /home/levi/master_project/Dsfeat/tools/run_dsfeat.py --cfg /home/levi/master_project/Dsfeat/experiments/tum.yaml


# hfnet
rm -r /home/levi/master_project/datasets_tum/results/hfnet_results/
mkdir -p /home/levi/master_project/datasets_tum/results/hfnet_results/

python3 /home/levi/master_project/hfnet/run_hfnet.py --image_folder /home/levi/master_project/datasets_tum/rgbd_dataset_freiburg3_walking_halfsphere/rgb/ \
--out_folder /home/levi/master_project/datasets_tum/results/hfnet_results/

