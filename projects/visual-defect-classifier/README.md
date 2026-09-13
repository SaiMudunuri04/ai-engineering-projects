# Visual defect classifier · deep learning

A PyTorch convolutional classifier for labelled inspection images. It checks that training and validation contain the same classes and rejects identical image files across splits. Training uses augmentation only on training images, AdamW, validation loss, early stopping, and a checkpoint with class names and seed.

Install with `python3 -m pip install -e '.[vision]'`. Arrange your own images as `dataset/train/good/*.png`, `dataset/train/defect/*.png`, `dataset/val/good/*.png`, and `dataset/val/defect/*.png`. Then run `python3 -m portfolio_ai.deep_learning dataset --output artifacts/vision-model.pt`. The sibling JSON report records validation history.

No proprietary or third-party image set is bundled. For production evaluation, split by physical asset or inspection batch as well as by file hash, and evaluate on a later unseen batch. A trained checkpoint and measured accuracy are not claimed until the workflow has been run on a suitable dataset.
