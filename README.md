
# Already registered data

Files _data/imaging_data/(256, 256, 156)_ and _data/imaging_data/(256, 256, 192)_ contain already the registered images (prefix _wm_)


# Data harmonization

## Running ComBat [NeuroHarmonize](https://github.com/rpomponio/neuroHarmonize)

Data must be registered and there should exist a CSV file with all the paths of the registered images. _registered_files.csv_ contains the paths. Training can be done using the following command (e.g., for the images in _data/imaging_data/(256, 256, 192)_): 

```
python3 train_neuroharmonize.py --file "data/imaging_data/(256, 256, 192)" --output_file "experiments/imaging_data/(256, 256, 192)"
```

In the folder _experiments/imaging_data/(256, 256, 192)/NeuroHarmonizer_X/images_ you can then find the produced MRIs.

## Running CatHarm

CatHarm consists of two steps:
1. Train the autoencoder
2. Generate the images

### Training the autoencoder

```
python3 train_images.py --file "data/imaging_data/(256, 256, 192)" --output_file "experiments/imaging_data/(256, 256, 192)" 
```

### Generating the images

```
python3 predict_images.py --image_file "data/imaging_data/(256, 256, 192)/registered_files.csv" --output_file "experiments/imaging_data/(256, 256, 192)/CatHarm_X" 
```
In the folder _experiments/imaging_data/(256, 256, 192)/CatHarm_X/images_ you can then find the produced MRIs.


# Predicting metadata

One way to evaluate the quality of the images (before and after harmonization) is by predicting the site-specific metadata (e.g., scanner type) as well as the _important_ metadata (e.g., diagnosis). This can be done as follows:

```
python3 eval_images.py --file "experiments/imaging_data/(256, 256, 192)/CatHarm_X"
```

