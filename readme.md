# EDA
- TMA resolution x40, WSI resolution x20 (10000x10000 ~ 100000x100000 resolution)
- we have only 25 TMA images with about 3000x3000 ~ 4000x4000 resolution

# My Approach
- TMA : center crop and tiling and inference with TMA model 
- WSI : first classify tumor area with WSI model and make further classify subtypes with TMA model

# TMA pipelne detail
- Train
    - extract tiles from WSI images with supplemental masks

- Inference
    - cropped and resized TMA tiles (extract tiles from TMA images with 2048x2048 resolution, and resize them to 512x512, stride 256, zoom : x40->x10)
    - inference with TMA model -> each tile with predicted ovarian sub type
    - majority votes and make final prediction

# WSI pipeline detail
- Train
    - Tumor classifier : TMA model but with WSI thumbnails and compressed WSI supplemental masks
    - TMA model : used the same TMA model from TMA pipeline

- Inference
    - tile WSI thumbnail image
    - inference with Tumor classifier -> each thumbnail tile with tumor or non-tumor result
        - no tumor tiles -> 'Other'
        - tumor tiles -> center crop and pass it to TMA model and do majority votes and make final prediction

# Reproduce Steps 
- 1. download images from https://www.kaggle.com/competitions/UBC-OCEAN/data
- 2. download supplemental masks from https://www.kaggle.com/datasets/sohier/ubc-ovarian-cancer-competition-supplemental-masks
(for TMA pipeline)
- 3. extract tiles from WSI from 1,2 data ->  v1 data (e.g. now we have "train_tiles_based_on_mask_1024x1024")
- 4. train first phase tma model with v1 data
- 5. extract more tiles from rest of the WSI images (WSI images without supplemental masks) -> v2 data
- 6. further train TMA model with v2 data

(for WSI pipeline)
- 7. extract thumbnail tiles from WSI from 1,2 data ->  v3 data 
- 8. train tumor classifier with v3 data

# Prepare data
### make v1 data
run `tma_1_decompose_wsi_tiles.py` script with below code
```python
extract_tiles_from_wsi_images_with_mask(image_dir, mask_dir, tile_size=(1024, 1024), stride=512, resize=(512, 512))
```

### make v2 data
- run `tma_2_inference_wsi_tiles.py` script with the TMA model from first phase
- run `tma_3_eda_inferenced_wsi_tiles_with_others.py` to select grouped tiles from above result

### make v3 data
run `tma_1_decompose_wsi_tiles.py` script with below code
```python
mask_dir = "/kaggle/working/UBC-OCEAN-compressed_supplemental_masks"
image_dir = "/kaggle/input/UBC-OCEAN/train_thumbnails"
extract_tiles_from_wsi_thumbnail_with_mask(
    image_dir,
    mask_dir,
    tile_size=(200, 200),
    stride=100,
    resize=(200, 200),
    dataset_prefix="train_thumbnail_based_on_mask",
)
```


# Train

### TMA model
- used to predict TMA images
- used to predict WSI tiles in WSI inference 2nd phase

##### Train
```bash
# use WSI masked selected tiles (about 100,000 images) for training
# use TMA images for validation (about 25 images, but will be centercrop to 100 images)
python3 train_tma.py --config config/train_tma_model_val_tma_only.yaml

```
##### Inference
```bash
# control Config class on top of the file
python3 inference.py

```


### WSI model
- used to predict WSI tiles in WSI inference 1st phase (to predict if WSI thumbnail tiles are cancerous or not)

##### Train
```bash
python3 train_wsi.py --config config/train_wsi_thumbnail_tile_model.yaml
```

##### Inference
```bash
# control Config class on top of the file
python3 inference.py

```
