
import os
import logging
import json
import argparse

from tqdm import tqdm
import dbdicom as db
import numpy as np

from utils.fatwatermap import fatwater
from utils.plot import volume_to_mosaic


def run(build, n=1):

    data = os.path.join(build, 'dixon', 'stage_2_data')
    model = os.path.join(build, 'fatwater', 'stage_6_build_distribution', 'FatWaterPredictor')
    output = os.path.join(build, 'fatwater', 'stage_7_test')

    # Get water series
    series = db.series(data)
    series_water = [s for s in series if s[3][0][-5:]=='water']

    # Loop over the water series
    for series_wi in tqdm(series_water[:n], desc='Writing training data'):

        # Patient and output study
        patient = series_wi[1]
        study = series_wi[2][0]
        series_wi_desc = series_wi[3][0]
        sequence = series_wi_desc[:-6] # remove '_water' suffix

        # Get fat/out_phase/in_phase series
        series_fi = series_wi[:3] + [(f'{sequence}_fat', 0)]
        series_op = series_wi[:3] + [(f'{sequence}_out_phase', 0)]
        series_ip = series_wi[:3] + [(f'{sequence}_in_phase', 0)]

        #
        # Compare original water map against reconstructed water map without T2* correction
        # 

        # Get out_phase/in_phase pixel data
        vol_op = db.volume(series_op)
        vol_ip = db.volume(series_ip)

        # Compute fat/water map without T2* correction
        f_rec, w_rec = fatwater(model, vol_op.values, vol_ip.values) 

        # Get original fat and water data for reference
        f_orig = db.volume(series_fi).values
        w_orig = db.volume(series_wi).values

        print('Fat image reconstruction error:', 100 * np.linalg.norm(f_rec-f_orig) / np.linalg.norm(f_orig))
        print('Water image reconstruction error:', 100 * np.linalg.norm(w_rec-w_orig) / np.linalg.norm(w_orig))

        # Save original + reconstructed water map in build folder
        os.makedirs(output, exist_ok=True)
        fname = f'{patient}__{study}__{sequence}'
        w_clip = [0, 0.6*min([w_rec.max(), w_orig.max()])]
        volume_to_mosaic(w_rec, save_as=os.path.join(output, f'{fname}__water_recon.png'), clip=w_clip)
        volume_to_mosaic(w_orig, save_as=os.path.join(output, f'{fname}__water.png'), clip=w_clip)
        f_clip = [0, 0.6*min([f_rec.max(), f_orig.max()])]
        volume_to_mosaic(f_rec, save_as=os.path.join(output, f'{fname}__fat_recon.png'), clip=f_clip)
        volume_to_mosaic(f_orig, save_as=os.path.join(output, f'{fname}__fat.png'), clip=f_clip)

        #
        # Compare original pddf against reconstructed pdff with T2* correction
        #

        # Get out_phase/in_phase echo times
        te_o = db.unique('EchoTime', series_op)[0]
        te_i = db.unique('EchoTime', series_ip)[0]
        
        # Compute fat fraction with T2* correction
        f_rec, w_rec = fatwater(model, vol_op.values, vol_ip.values, te_o, te_i) 

        pdff_rec = np.zeros_like(f_rec, dtype=float)
        signal = f_rec + w_rec
        np.divide(f_rec, signal, out=pdff_rec, where=signal!=0)

        # Compare to fat fraction with original fat-water
        pdff_orig = np.zeros_like(f_orig, dtype=float)
        signal = f_orig + w_orig
        np.divide(f_orig, signal, out=pdff_orig, where=signal!=0)

        # Save pddf maps
        volume_to_mosaic(pdff_rec, save_as=os.path.join(output, f'{fname}__pdff_recon.png'), clip=[0,1])
        volume_to_mosaic(pdff_orig, save_as=os.path.join(output, f'{fname}__pdff.png'), clip=[0,1])



if __name__=='__main__':

    BUILD = r'C:\Users\md1spsx\Documents\Data\iBEAt_Build'
    os.makedirs(BUILD, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(BUILD, 'error.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    parser.add_argument("--n", type=int, default=1, help="Build folder")
    args = parser.parse_args()

    run(args.build, n=args.n)