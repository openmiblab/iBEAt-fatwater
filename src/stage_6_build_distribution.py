import os
import shutil
import subprocess
import argparse


def run(build):

    # Build a lightweight folder with bare model weights for distribution.
    source = os.path.join(build, 'fatwater', 'stage_4_train', 'nnUNet_results', 'Dataset011_iBEAtFatWater', 'nnUNetTrainer__nnUNetPlans__3d_fullres')

    # Destination folder
    destination = os.path.join(build, 'fatwater', 'stage_6_build_distribution', 'FatWaterPredictor', 'Dataset001_FatWaterPredictor', 'nnUNetTrainer__nnUNetPlans__3d_fullres')
    os.makedirs(destination, exist_ok=True)

    # Copy required header files
    file = os.path.join(source, 'dataset.json')
    shutil.copy(file, destination)
    file = os.path.join(source, 'dataset_fingerprint.json')
    shutil.copy(file, destination)
    file = os.path.join(source, 'plans.json')
    shutil.copy(file, destination)

    # copy weights
    for fold in [0, 1, 2, 3, 4]:
        file = os.path.join(source, f'fold_{fold}', 'checkpoint_final.pth')
        fold_dest = os.path.join(destination, f'fold_{fold}')
        os.makedirs(fold_dest, exist_ok=True)
        shutil.copy(file, fold_dest)

    # Files needed for postprocessing
    source = os.path.join(source, "crossval_results_folds_0_1_2_3_4")
    destination = os.path.join(destination, "crossval_results_folds_0_1_2_3_4")
    os.makedirs(destination, exist_ok=True)
    file = os.path.join(source, 'postprocessing.pkl')
    shutil.copy(file, destination)
    file = os.path.join(source, 'plans.json')
    shutil.copy(file, destination)


if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    # Comment for the cluster
    os.environ['nnUNet_n_proc_DA'] = '4' # Set in .sh file
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)


