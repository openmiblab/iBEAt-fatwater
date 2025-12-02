# Model training for fat-water mapping from 3D Dixon-MRI

[![Code License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square&logo=apache&color=blue)](https://www.apache.org/licenses/LICENSE-2.0) [![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/) [![Output Data](https://img.shields.io/badge/output%20data-Zenodo-FF8C00?logo=databricks&logoColor=white)](https://doi.org/10.5281/zenodo.17791059)


## 📚 Background

Computation of fat and water images from a 2-point MRI Dixon acquisition is usually done in-line by the scanner software, and requires access to the phase and magnitude data. 

In some cases one may want to compute fat and water images retrospectively - for instance when they were not originally exported, or in order to reconstruct them with different models (e.g. with correction for T2* decay, B0-effects, etc). This causes a practical problem when, as is common, phase images or not stored and only magnitude images of in-phase and opposed-phase scans are available. 

The crucial bit of information that is missing with magnitude-only data is the sign of the opposed phase image - does the pixel contain mostly water or mostly fat? This pipeline trains a deep learning model to recover this binary information from magnitude images of in-phase and opposed-phase data. 


## 🛠️ Methods

Training data are taken from 1,143 3D Dixon scans of the abdomen acquired in the [iBEAt study on diabetic kidney disease](https://pubmed.ncbi.nlm.nih.gov/32600374/). The dataset includes pre- and post contrast agent images from patients, and precontrast scans in volunteers. Data were acquired with a field of view of 400 mm in both the read and phase directions, a slice thickness of 1.5 mm, and 144 slices per slab. The repetition time (TR) was 4.01 ms, with two echo times (TE) at 1.34 ms and 2.57 ms. 

The pipeline trains an [nUNet model](https://github.com/MIC-DKFZ/nnUNet) to predict a binary image with value=1 in pixels that contain mostly water, and 0 otherwise. A two-channel input is used accepting in-phase and opposed-phase magnitude images. 


## 📁 Code structure

This `src` folder contains all the steps needed to reproduce the trained model. As the iBEAt source data are currently still embargoed, the source data are only available to `miblab` members. The pipeline itself is split up into 8 consecutive stages. Each stage will produce a subfolder in a user-defined `build` folder. Subfolders are named after the stage of the pipeline that has produced them.

- `stage_0_restore data.py` restores the data from the `miblab` google drive archive to a local disk and can only be run by `miblab` members with access to the google drive. The output of this stage is a folder `data\dixon` with Dixon data in DICOM format.
- `stage_1_compute_labels.py` derives the binary ground-truth images from the available fat and water data. The labels are saved in DICOM in a folder `build\stage_1_labels`.
- `stage_2_training_data.py` organises the orginal Dixon source images (in and opposed-phase) and computed labels into the format required by nnUNet for training. The results are stored in the folder `build\stage_2_training_data`.
- `stage_3_preprocess.py` runs the nnUNet preprocessing step on the training data and saves the results in `build\stage_3_preprocess`.
- `stage_4_train.py` performs the actual training, fold-by-fold (5 folds in total). All results (trained model weights, logs and validation results) are saved in `build\stage_4_train`.
- `stage_5_find_config.py` uses standard nnUNet functions to find the optimal configuration. The results are also saved in `build\stage_4_train`.
- `stage_6_build_distribution` takes the trained model weights and essential json files to construct a light-weight distribution folder `build\stage_6_distribution`. 
- `stage_7_test.py` performs a sanity check on the python API wrapper in `src\utils\fatwatermap.py` by reconstructing fat and water images for a few cases, exporting the 3D volumes as mosaics, and comparing the result against fat and water maps reconstructed on the scanner. 
 

## 💻 Running the pipeline

To run the pipeline, first install the conda environment and activate it:

```bash
conda env create -n fatwater -f environment.yml
conda activate fatwater
```

In theory the complete pipeline can be run in one go from the terminal, using the top-level function `pipeline.py` and paths to archive and build folders as arguments:

```bash
python src/pipeline.py --archive=path/to/archive --build=path/to/output
```

In practice we have run the stages one by one, with all preprocessing run locally on a laptop PC:

```bash
python src/stage_0_restore_data.py --archive=path/to/archive --build=path/to/output
python src/stage_1_compute_labels.py --build=path/to/output
python src/stage_2_training_data.py --build=path/to/output
python src/stage_3_preprocess.py --build=path/to/output
```

The model training was run on the high-performance cluster (HPC) of Sheffield University. See the [miblab tutorial](https://github.com/openmiblab/tutorial-cluster) for more detail on how to run jobs on the HPC. One job was submitted for each fold, with a run time of about 2 days for each of the 5 folds. The script used to submit the jobs is included in this distribution. It was run 5 times after changing the `fold` argument in the last line:

```bash
sbatch hpc/stage_3_train.sh
```

After training on the HPC, the results were pulled back and the final steps were again run locally:

```bash
python src/stage_5_find_config.py --build=path/to/output
python src/stage_6_build_distribution.py --build=path/to/output
python src/stage_7_test.py --build=path/to/output
```

The end result for the model weights has been uploaded manually to [Zenodo](https://zenodo.org/records/17791059). 


## 📦 Using the trained model

The purpose of the model is to enable a reconstruction of fat and water images from magnitude-only in-phase and opposed-phase images. This functionality is available via the `fatwater()` function in the python package `miblab-dl`, which uses the model prediction together with source images to produce the output.

In order to run it, first  `pip install miblab-dl` in an environment with a suitable `pytorch` installation. After that, fat and water can be computed from numpy arrays containing in- and opposed-phase images:

```python
from miblab_dl import fatwater

fat_image, water_image = fatwater(
    opposed_phase_image,    # 3D numpy array (columns, rows, slices)
    in_phase_image,         # 3D numpy array (same shape)
)
```
Under the hood this function uses the trained model to generate the binary sign image, and then combines that with the source images to compute fat and water. Correction for T2* decay can be built in as an option:

```python
fat_image, water_image = fatwater(
    opposed_phase_image,        # 3D numpy array (columns, rows, slices)
    in_phase_image,             # 3D numpy array (same shape)
    opposed_phase_echo_time,    # opposed-phase TE in ms
    in_phase_echo_time,         # in-phase TE in ms
    t2star_water,               # T2* of water in ms     
    t2star_fat,                 # T2* of fat in ms
)
```

## 💰 Funding 

The work was performed as part of the [BEAt-DKD project](https://www.beat-dkd.eu/) on biomarkers for diabetic kidney disease. The project was EU-funded through the [Innovative Health Initiative](https://www.ihi.europa.eu/).


## 👥 Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JoaoPeriquito"><img src="https://avatars.githubusercontent.com/u/48806417?v=4" width="100px;" alt="Joao Periquito"/><br /><sub><b>Joao Periquito</b></sub></a><br /></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/plaresmedima"><img src="https://avatars.githubusercontent.com/u/6051075?v=4" width="100px;" alt="Steven Sourbron"/><br /><sub><b>Steven Sourbron</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
