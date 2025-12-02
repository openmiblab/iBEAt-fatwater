import os
import logging
import argparse

import dbdicom as db


def run(archive, local):
    run_batch(archive, local, 'Controls')
    for site in ['Bordeaux', 'Bari', 'Leeds', 'Sheffield', 'Turku', 'Exeter']:
        run_batch(archive, local, 'Patients', site)


def run_batch(remotepath, localpath, group, site=None):
    datapath = os.path.join(localpath, 'dixon', 'stage_2_data')
    archivepath = os.path.join(remotepath, "dixon", "stage_2_data")
    if group == 'Controls':
        sitedatapath = os.path.join(datapath, 'Controls')
        sitearchivepath = os.path.join(archivepath, 'Controls')
    else:
        sitedatapath = os.path.join(datapath, 'Patients', site)
        sitearchivepath = os.path.join(archivepath, 'Patients', site)
    db.restore(sitearchivepath, sitedatapath)


if __name__=='__main__':

    ARCHIVEPATH = r'G:\Shared drives\iBEAt_Build'
    LOCALPATH = r'C:\Users\md1spsx\Documents\Data\iBEAt_Build'

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str, default=ARCHIVEPATH, help="Build folder")
    parser.add_argument("--build", type=str, default=LOCALPATH, help="Build folder")
    args = parser.parse_args()

    run(args.archive, args.build)

