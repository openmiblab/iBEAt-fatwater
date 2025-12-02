import argparse

import stage_0_restore_data
import stage_1_compute_labels
import stage_2_training_data
import stage_3_preprocess
import stage_4_train
import stage_5_find_config
import stage_6_build_distribution
import stage_7_test


if __name__=='__main__':

    ARCHIVEPATH = r'G:\Shared drives\iBEAt_Build'
    LOCALPATH = r'C:\Users\md1spsx\Documents\Data\iBEAt_Build'

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str, default=ARCHIVEPATH, help="Build folder")
    parser.add_argument("--build", type=str, default=LOCALPATH, help="Build folder")
    args = parser.parse_args()

    stage_0_restore_data.run(args.archive, args.build)
    stage_1_compute_labels.run(args.build) 
    stage_2_training_data.run(args.build)
    stage_3_preprocess.run(args.build)
    stage_4_train.run(args.build, fold=0)
    stage_4_train.run(args.build, fold=1)
    stage_4_train.run(args.build, fold=2)
    stage_4_train.run(args.build, fold=3)
    stage_4_train.run(args.build, fold=4)
    stage_5_find_config.run(args.build)
    stage_6_build_distribution.run(args.build)
    stage_7_test(args.build, n=1)
