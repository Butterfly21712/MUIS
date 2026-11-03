# MUIS

MUIS combines XFeat and LighterGlue feature matching, bounded geometric optimization with TRF, and learned composition for multi-image stitching. The repository includes the pretrained composition checkpoint `epoch050_newmodel.pth` for inference and supports training from scratch.

Install the environment from the project root:

```bash
conda env create -f environment.yml
conda activate muis
python -m pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

Run all commands below from the project root. `main.py` is the main entry point for multi-image stitching; other entry points are invoked with `python -m`.

## 1. Using the Pretrained Checkpoint

### Download feature matching weights

```bash
python -c "from pathlib import Path; Path('weights').mkdir(exist_ok=True)"
python -c "from urllib.request import urlretrieve; urlretrieve('https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt', 'weights/xfeat.pt')"
python -c "from urllib.request import urlretrieve; urlretrieve('https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt', 'weights/xfeat-lighterglue.pt')"
```

Arrange the weights as follows:

```text
MUIS_Method/
├── epoch050_newmodel.pth
└── weights/
    ├── xfeat.pt
    └── xfeat-lighterglue.pt
```

### Stitch your own images

Place images from one sequence in `data/demo/`, named in capture order, such as `001.jpg`, `002.jpg`, and `003.jpg`. Adjacent images should overlap. JPG, JPEG, and PNG files are supported.

```bash
python main.py --input_dir data/demo --checkpoint epoch050_newmodel.pth --output results/panorama.jpg
```

The script automatically determines the image count and processes images in natural filename order. The panorama is saved to `results/panorama.jpg`. Images are resized to 640×360 by default; use `--width` and `--height` to change the input size.

### Run on the UDIS-D test set

Download UDIS-D from [Google Drive](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/13KZ29e487datgtMgmb9laQ) (access code: `1234`). Arrange the test images as follows:

```text
data/UDIS-D/testing/
├── input1/
└── input2/
```

Images in the two directories must have matching filenames. Generate the aligned images and masks, then run composition:

```bash
python -m datasets.prepare --input_dir data/UDIS-D/testing --output_dir data/MUIS/testing --gpu 0
python -m inference.dataset --test_path data/MUIS/testing --checkpoint epoch050_newmodel.pth --output_dir results/pretrained --gpu 0 --batch_size 1
```

Composed images are saved to `results/pretrained/composition/`. Predicted masks are saved to `results/pretrained/mask1/` and `results/pretrained/mask2/`.

## 2. Training from Scratch

### Prepare the training data

Complete the environment setup and feature weight downloads above. Preserve the official UDIS-D training/test split:

```text
data/UDIS-D/
├── training/
│   ├── input1/
│   └── input2/
└── testing/
    ├── input1/
    └── input2/
```

Generate training inputs using our method:

```bash
python -m datasets.prepare --input_dir data/UDIS-D/training --output_dir data/MUIS/training --gpu 0
```

The generated directory contains `warp1/`, `warp2/`, `mask1/`, and `mask2/`. Prepare the test set once using the commands in Section 1; it does not need to be generated again.

### Train the composition network

```bash
python -m training.train --train_path data/MUIS/training --gpu 0 --batch_size 1 --max_epoch 50 --learning_rate 0.0001 --num_workers 0 --seed 42 --output_dir runs/composition
```

This command randomly initializes the composition network without loading the supplied composition checkpoint. Training uses Adam, with the learning rate multiplied by 0.97 after each epoch. Checkpoints are saved every 10 epochs and at the end of training to `runs/composition/checkpoints/`.

View training logs:

```bash
tensorboard --logdir runs/composition/logs
```

Resume from a checkpoint saved during training:

```bash
python -m training.train --train_path data/MUIS/training --gpu 0 --max_epoch 50 --output_dir runs/composition --resume runs/composition/checkpoints/epoch020_newmodel.pth
```

`--max_epoch` specifies the target total number of epochs. Use a new `--output_dir` to start another training run.

### Evaluate the trained model

```bash
python -m inference.dataset --test_path data/MUIS/testing --checkpoint runs/composition/checkpoints/epoch050_newmodel.pth --output_dir results/trained --gpu 0 --batch_size 1
```

Stitch your own images using the newly trained model:

```bash
python main.py --input_dir data/demo --checkpoint runs/composition/checkpoints/epoch050_newmodel.pth --output results/trained_panorama.jpg
```

## Project Structure

```text
MUIS_Method/
├── main.py                  # Main entry point for multi-image stitching
├── epoch050_newmodel.pth     # Pretrained composition checkpoint
├── geometry/                # TRF camera estimation and coordinate projection
├── features/                # XFeat and LighterGlue feature matching
├── composition/             # Composition network, losses, and model loading
├── datasets/                # Data loading and preprocessing
├── training/                # Training entry point
├── inference/               # Dataset and image-pair inference
├── evaluation/              # Alignment metrics
├── weights/                 # Feature matching weights (created after download)
└── README.md
```
