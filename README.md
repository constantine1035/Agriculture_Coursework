6.8 Segmentation on the Agriculture Vision Dataset  
6.8.1 Dataset Preparation  

**Dataset structure:**  
```
train/
 ├─ images/
 │   ├─ rgb/    (*.jpg)
 │   └─ nir/    (*.jpg)
 ├─ labels/
 │   ├─ double_plant/       (*.png)
 │   ├─ drydown/            (*.png)
 │   …  
 │   └─ weed_cluster/       (*.png)
 └─ masks/                   (*.png)
```

1. **Merging separate masks into a single multiclass mask**  
   To train segmentation models from the SegmentationModelsPyTorch library [32] on a unified multiclass format, we implemented a script that:
   1. Loads all binary masks for each class (e.g. double_plant, …, weed_cluster), where pixels of that class are 255 and all others are 0.  
   2. Assigns each class a unique index:
      ```python
      labels_to_index = {
        "double_plant": 1,
        "drydown": 2,
        …,
        "weed_cluster": 9
      }
      ```
      The background remains index 0.  
   3. Overlays the masks in ascending class-index order. For each class, pixels are set to its index; if classes overlap, the higher index wins.  
   4. Saves the combined mask as `masks/<image_id>.png`, where each pixel stores its integer class.  

   The result is a single multiclass mask per image, suitable for SMP models and for visualizing the ground truth.

2. **Generating text annotations for YOLO-seg**  
   Segmentation versions of YOLO expect, for each image, a text file rather than a PNG mask. Each line has the format:
   ```
   <class_idx> x1 y1 x2 y2 … xN yN
   ```
   where the (xi, yi) are the normalized coordinates of the region’s polygon. Our converter function does the following for each image:
   1. Opens the RGB file and reads its width and height.  
   2. Iterates over `labels_root/<class_name>/<image_id>.png`.  
   3. For each binary mask:
      - Reads it in gray scale (`cv2.IMREAD_GRAYSCALE`).  
      - Finds external contours with `cv2.findContours`.  
      - Converts each contour to a list of (x, y) points and normalizes them by the image dimensions.  
      - Assembles a line `class_idx x1_norm y1_norm … xK_norm yK_norm`.  
   4. Writes all lines to a TXT file in the output directory.  

   These TXT annotations are then passed to `model.train(..., data="dataset.yaml", ...)`.

   By (1) combining PNG masks into a multiclass map and (2) converting it to the YOLO-seg format, we can:
   - Train SMP models on nine classes plus background.  
   - Train YOLO-seg on the same data, with each region as a polygon.  

   Both pipelines use identical validation annotations, enabling a fair SMP vs. YOLO-seg comparison.

3. **Class balancing and sample selection**  
   An initial metadata inspection revealed that the background and some anomalies (weed_cluster, double_plant) were tens of times more frequent than rare classes (storm_damage, planter_skip).

   _Figure 86. Histogram of class distribution in train and validation before balancing_  

   To mitigate bias toward frequent classes, we applied:
   - **Metadata collection:** for each YOLO TXT file, record which classes appear.  
   - **Iterative removal:** until enough pairs were removed, repeat:
     1. Recompute remaining class frequencies.  
     2. Identify the top three most frequent classes.  
     3. Remove the first file whose classes are all within that top-3 set.  

   **Results:**  
   - Removed from training: 36 944 files (≈ 65 percent of the original train set)  
   - Retained in training: 20 000 files  
   - Removed from validation: 13 334 files  
   - Retained in validation: 5 000 files  

   A small imbalance remains, but it is greatly improved.

   **Class distribution after balancing:**  
   - Training: {0: 4691, 1: 4521, 2: 4384, 3: 4576, 4: 4285, 5: 934, 6: 2732, 7: 3337, 8: 8880}  
   - Validation: {0: 1045, 1: 1549, 2: 940, 3: 1026, 4: 1343, 5: 122, 6: 860, 7: 702, 8: 1248}  

   _Figure 87. Histogram of class distribution in train and validation after balancing_  

---

6.8.2 Training YOLO Models  

This section covers data configuration, model setup, training, and metric analysis for YOLO segmentation.

1. **Dataset configuration**  
   In `dataset.yaml`, specify:
   - The dataset root path  
   - Class names  
   - Number of classes  
   - Relative paths to training and validation image lists  

2. **Annotation directories**  
   - `train/labelsYOLO` and `val/labelsYOLO` each contain one TXT file per image, generated as above.  

3. **Model and hyperparameter selection**  
   We evaluated five YOLOv11-seg variants:
   - YOLO11n-seg (nano)  
   - YOLO11s-seg (small)  
   - YOLO11m-seg (medium)  
   - YOLO11l-seg (large)  
   - YOLO11x-seg (extra-large)  

   **Common training settings:**  
   - Optimizer: AdamW  
   - Initial learning rate: 0.01 for nano and small, 0.005 for medium and large, 0.002 for extra-large  
   - Batch size: 64 for the first three; 32 for large and extra-large  
   - Image size: 512 × 512  
   - Epochs: up to 25 with early stopping (patience = 5)  
   - Device: NVIDIA RTX 3090 with automatic mixed precision  

   **Example command for YOLO11s-seg:**
   ```python
   model = YOLO("yolo11s-seg.pt")
   model.train(
       data="dataset.yaml",
       epochs=25,
       imgsz=512,
       batch=64,
       optimizer="AdamW",
       lr0=1e-2,
       lrf=0.01,
       project="YOLOSEG",
       name="YOLO11s_seg",
       device="0",
       seed=42,
       patience=5
   )
   ```

4. **Collection and interpretation of metrics**  
   1. **Log unification:** each model’s `results.csv` holds per-epoch  
      - Losses: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `train/seg_loss` and their validation counterparts  
      - Metrics: `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`  
      We loaded them into pandas DataFrames.  
   2. **Loss curves:** on a 4 × 2 grid, plot training vs. validation for box loss, segmentation loss, classification loss, and distribution focal loss.  
      _Figure 88. Loss functions_  
   3. **Key metrics:** on a 2 × 2 grid, plot precision, recall, mAP@0.5, and mAP@0.5–0.95 over epochs for all five models.  
      _Figure 89. Key metrics_  
   4. **Detailed artifacts:** save and display F1 and precision–recall curves (e.g., `MaskF1_curve.png`, `MaskPR_curve.png`) and confusion matrices (`confusion_matrix.png`).  
      _Figure 90. Mask F1 and confusion matrix_  
      These visualizations reveal common class confusions and guide threshold selection.  
      _Figure 91. Mask F1 and confusion matrix_  

5. **Key conclusions:**  
   - All models exhibit a sharp drop in box and segmentation losses during the first 5–10 epochs, then level off.  
   - Nano and small models reach ~0.30–0.32 precision by epoch 10 but lag in mAP and F1.  
   - The medium model offers the best trade-off, with the smallest train/validation gap.  
   - The extra-large model attains the highest final mAP@0.5 (~0.33) and mAP@0.5–0.95 (~0.17), but only marginally better than large and medium.  
   - Based on convergence speed, validation stability, and final metrics, YOLO11m-seg (medium) was chosen as the optimal compromise.  
   - Training times (hours): nano 0.979, small 1.212, medium 3.242, large 2.320, extra-large 3.650. The medium model’s longer time is likely an artifact.

6. **Practical recommendations:**  
   - **Augmentations:** keep Mosaic, MixUp, HSV, and RandomFlip enabled for better generalization.  
   - **Automatic mixed precision:** `amp=True` gives ~1.5× speedup on RTX 3090 with no accuracy loss.  
   - **Batch size and learning rate:** lower the learning rate and batch size for larger models to optimize memory usage.

---

6.8.3 Training SMP Models  
This subsection details the pipeline for segmentation_models_pytorch (SMP).

1. **Data preparation**  
   - Use `AgricultureVisionDataset`, subclassed from `torch.utils.data.Dataset`.  
   - Define transforms in `get_transforms(image_size)`:

     **Training:**  
     - A.Resize(*image_size)  
     - A.HorizontalFlip(p=0.5)  
     - A.VerticalFlip(p=0.5)  
     - A.RandomRotate90(p=0.5)  
     - A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)  
     - A.Normalize(mean, std)  
     - ToTensorV2()  

     **Validation:**  
     - A.Resize(*image_size)  
     - A.Normalize(mean, std)  
     - ToTensorV2()  

   - `get_dataloaders` produces `DataLoader` objects for train and val, with `batch_size` and `num_workers` tuned to hardware.

2. **Model construction**  
   - Use `create_model(model_name, encoder_name, num_classes, in_channels=3, encoder_weights="imagenet")`.  
   - Encoders are initialized with pretrained weights.

3. **Training and validation**  
   - `train_model` loop per epoch:
     1. Compute and log pixel-averaged segmentation loss.  
     2. Call `evaluate_epoch_metrics` to obtain:
        ```json
        {
          "seg_loss": …,
          "precision": …,
          "recall": …,
          "f1": …,
          "conf_mat": …
        }
        ```
     3. Append metrics to `history` and save as `history_{model_name}.csv`.

   _Figures 92–96. Loss decrease per epoch for U-Net, U-Net++, FPN, DeepLabV3, DeepLabV3Plus_

4. **Results analysis**  
   | Model                                | val_seg_loss        | val_precision        | val_recall         | val_f1               |
|--------------------------------------|---------------------|----------------------|--------------------|----------------------|
| FPN_timm-efficientnet-b4             | 0.6122031562060120  | 0.5594803780066290   | 0.408288862865197  | 0.4454766556535360   |
| DeepLabV3_resnet34                   | 0.6270773340940480  | 0.5288257867030410   | 0.3811813832352850 | 0.4104384411469280   |
| Unet_timm-efficientnet-b4            | 0.6333322314351800  | 0.5487992858091560   | 0.3362768231998710 | 0.3807695830895140   |
| FPN_resnet34                         | 0.6606511295124890  | 0.5378097759781410   | 0.3344164679422720 | 0.3746928298838060   |
| UnetPlusPlus_timm-efficientnet-b4   | 0.5775481132000690  | 0.5947141013120660   | 0.3190662424282850 | 0.3667029596818100   |
| DeepLabV3Plus_timm-efficientnet-b4  | 0.8851941756725310  | 0.4934078283839480   | 0.3381729246357480 | 0.3627181760821130   |
| FPN_resnet101                        | 0.6457951766252520  | 0.5797755830359630   | 0.3205214881408680 | 0.3447069711588920   |
| UnetPlusPlus_resnet101               | 0.7171722949624060  | 0.4575020942029130   | 0.2513185247063580 | 0.2680046603658110   |
| Unet_resnet34                        | 0.7467995396912100  | 0.5190398528582190   | 0.2404703813230600 | 0.2589075418859800   |
| UnetPlusPlus_resnet34                | 0.7636833560407160  | 0.4799698662869120   | 0.2493336121208670 | 0.2532202080771880   |
| Unet_resnet101                       | 0.7806547465324400  | 0.3057655209811820   | 0.2402734257972700 | 0.2339638389063830   |
| DeepLabV3_timm-efficientnet-b4       | 0.8296669184193020  | 0.4031217705483470   | 0.2036448617064900 | 0.2207773370528010   |
| DeepLabV3Plus_resnext101_32x8d       | 1.1896200051754700  | 0.2937001021619760   | 0.1351138160711800 | 0.1174290746501570   |

   **Table 34.** Training results for various SMP models

   **Conclusions:**  
   1. FPN with EfficientNet-B4 achieves the best precision-recall balance (F1 ≈ 0.4455), indicating strong multiscale feature aggregation.  
   2. DeepLabV3 with ResNet-34 ranks second (F1 ≈ 0.4104).  
   3. U-Net with EfficientNet-B4 is third (F1 ≈ 0.3808), demonstrating the value of modern encoders.  
   4. U-Net++ (EfficientNet-B4) attains top loss and precision but lower recall.  
   5. DeepLabV3Plus with ResNeXt-101 performs worst (F1 ≈ 0.12), likely due to data and resource limitations.  

   Focus further experiments (augmentations, learning-rate schedules, regularization) on the top three SMP models:  
   **FPN (EfficientNet-B4) → DeepLabV3 (ResNet-34) → U-Net (EfficientNet-B4).**

---

6.8.4 YOLO vs. SMP Comparison  
To compare YOLO-seg and SMP models objectively, we built a script with these stages:

1. **Configuration:**  
   - Specify the class count.  
   - Set paths to validation images, SMP multiclass masks, and YOLO annotations.  
   - List models: five YOLO-seg variants and fifteen SMP checkpoints (UNet, FPN, DeepLabV3/Plus, U-Net++ with various encoders).

2. **Dataset preparation:**  
   - **SMPValDataset:** Resize to 512×512, Normalize, ToTensorV2 (albumentations).  
   - **YOLOValDataset:** Dynamically resize to 512×512 (square), preserve scale for output cropping, apply standard augmentations.

3. **Metrics:**  
   - `fast_confusion` builds a NUM_CLASSES × NUM_CLASSES confusion matrix.  
   - `compute_segmentation_metrics` computes per-class precision, recall, IoU, F1 and averages them.

4. **Evaluation:**  
   - `eval_smp_model`: loads each SMP checkpoint, predicts on validation, accumulates confusion matrix, returns metrics.  
   - `eval_yolo_model`: uses `ultralytics.YOLO.predict` to get binary masks, assigns class indices, crops to original size, updates confusion matrix.

5. **Results summary:**  
   | Model                               | mean_iou | mean_f1  | mean_precision | mean_recall |
|-------------------------------------|----------|----------|----------------|-------------|
| YOLOx                               | 0.332109 | 0.463627 | 0.488667       | 0.463709    |
| FPN_timm-efficientnet-b4.pth        | 0.325045 | 0.445198 | 0.542821       | 0.411758    |
| YOLOm                               | 0.324024 | 0.448173 | 0.491590       | 0.436494    |
| YOLOl                               | 0.317534 | 0.443035 | 0.506743       | 0.434688    |
| YOLOs                               | 0.315383 | 0.441214 | 0.501627       | 0.420038    |
| YOLOn                               | 0.310526 | 0.436422 | 0.492615       | 0.432873    |
| DeepLabV3Plus_resnet34              | 0.282345 | 0.390231 | 0.567223       | 0.354773    |
| DeepLabV3_resnet34                  | 0.277324 | 0.382511 | 0.542463       | 0.358309    |
| Unet_timm-efficientnet-b4           | 0.273629 | 0.380780 | 0.548775       | 0.336288    |
| UnetPlusPlus_timm-efficientnet-b4  | 0.269958 | 0.366725 | 0.594645       | 0.319084    |
| FPN_resnet34                        | 0.269327 | 0.374725 | 0.537787       | 0.334451    |
| FPN_resnet101                       | 0.244844 | 0.344673 | 0.579814       | 0.320499    |
| DeepLabV3_resnext101_32x8d          | 0.235812 | 0.333947 | 0.540217       | 0.320625    |
| DeepLabV3_timm-efficientnet-b4      | 0.212947 | 0.302213 | 0.440061       | 0.292611    |
| UnetPlusPlus_resnet34               | 0.197435 | 0.279858 | 0.532557       | 0.243245    |
| UnetPlusPlus_resnet101              | 0.188568 | 0.268028 | 0.457467       | 0.251342    |
| Unet_resnet34                       | 0.185756 | 0.258907 | 0.519058       | 0.240473    |
| Unet_resnet101                      | 0.170558 | 0.233951 | 0.305766       | 0.240257    |
| DeepLabV3Plus_resnext101_32x8d      | 0.095079 | 0.124385 | 0.325130       | 0.125463    |

   **Table 35.** Models sorted by descending mean IoU

   **Key findings:**  
   - **YOLOx** achieves the highest mean IoU (0.332), followed closely by **YOLOm** (0.324) and **FPN** (0.325).  
   - SMP models reach higher precision (up to 0.59) but lower recall (~0.32–0.42); YOLO models have higher recall (~0.42–0.46).  
   - Average F1 favors **YOLOx** (0.463), then **FPN** (0.445), **DeepLabV3Plus** (0.421), **U-Net** (0.381), etc.  

   **Conclusion:**  
   YOLOx provides the best balance between IoU, F1, and recall—crucial when missing problematic regions is costly. SMP models (notably FPN and U-Net++) excel in precision and can refine segment boundaries. Further exploration of SMP backbones may yield a model that outperforms YOLO while being more efficient to train.
