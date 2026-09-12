
<div align="center">
<h1>Understanding YOLO Training/Evaluation Results</h1>
<img width="1136" height="574" alt="nairolab-github-banner-cropped" src="https://github.com/NAIROlab-ke/Strathmore-School_ML-AI-Club_3rd-Term-2026/blob/main/nairolab-github-banner-cropped.png" />
</div>

## Table Of Contents

1. Intro/Background
2. Losses
3. Metrics
4. TL;DR
5. Practical tips when reading `results.png`
   
## Intro/Background

- The Google Colab Notebook training output is stored in the `MyDrive` directory of your Google Drive under the name `output-TIMESTAMP.zip` (where `TIMESTAMP` is the form `YearMonthDay_HourMinSec`).
- A **training/evaluation dashboard** for the YOLO instance segmentation model can be found in the file `train/content/runs/segment/train/results.png`. It contains two broad categories for analysis: **training/validation losses** and **validation performance metrics**
	
	* **Losses** → what the model is trying to minimize during training (i.e. the **error**). Calculated from **both training and validation data**
	* **Metrics** → how well the trained model actually performs on **validation data**

	After each epoch during YOLO training:
	```
	TRAINING
	  train images
	      ↓
	  forward pass → calculate losses
	      ↓
	  backpropagation → update weights
	      ↓
	VALIDATION
	  val images
	      ↓
	  make predictions
	      ↓
	  calculate metrics
	      ↓
	  precision / recall / mAP
	  ```
 
- Recall that the YOLO dataset is split into `train`, `val`, and `test` sections:
	- `train`: data/images actually used to **update the model weights** during training.
	- `val`: data/images held out from weight updates and used to **evaluate the model during training, and typically after each epoch**. These results help **monitor convergence/overfitting** and can be used for **model selection/checkpointing**.
	- `test`: a held-out set intended for **final, unbiased evaluation after training/model selection is finished.**

	```
	                    DATASET
	                       │
	          ┌────────────┼────────────┐
	          ↓            ↓            ↓
	       TRAIN          VAL          TEST
	          │            │            │
	          │            │            │
	     update weights   evaluate    final evaluation
	          │          every epoch    after training
	          │            │            │
	          └──────┬─────┘            │
	                 ↓                  │
	           choose/monitor           │
	             model                  │
	                 │                  │
	                 └──────────────────┘
	                         ↓
	                  FINAL TEST RESULT
	```
	
	where `choose/monitor model` --> `weights/{best.pt,last.pt}`
	
	```	
	TRAIN
	  ↓
	train model
	  ↓
	VAL ← use this to make training/model decisions
	  ↓
	freeze model choice
	  ↓
	TEST ← evaluate once
	```

 - **JUST SKIP TO SECTION [3. TL;DR](https://github.com/NAIROlab-ke/Strathmore-School_ML-AI-Club_3rd-Term-2026/blob/main/sep-10-comments.md#3-tldr) FOR A COMPACT SUMMARY OF SECTIONS [1. Losses](https://github.com/NAIROlab-ke/Strathmore-School_ML-AI-Club_3rd-Term-2026/blob/main/sep-10-comments.md#1-losses) AND [2. Metrics](https://github.com/NAIROlab-ke/Strathmore-School_ML-AI-Club_3rd-Term-2026/blob/main/sep-10-comments.md#2-metrics)**

## 1. Losses

### Overview

- **Lower is better**
- The train/evaluation relationship matters more than the absolute number.
	- `val/*` curves should generally **track** the `train/*` curves
	- For instance, if a `val` curve starts to rise while `train` continues to drop, then it is a sign of **overfitting**

### Loss Plots

- `train/box_loss` / `val/box_loss`: This measures how accurately YOLO predicts the bounding boxes around objects.
- `train/seg_loss` / `val/seg_loss`: This is the equivalent for the segmentation masks. measures how well the predicted object mask corresponds to the ground-truth mask. The bbox can be excellent while the mask is poor.
- `train/cls_loss` / `val/cls_loss`: This is the classification loss. Once YOLO has identified an object, it has to answer: *"What class is this?"*
- `train/dfl_loss` / `val/dfl_loss`: This one is more YOLO-specific. **DFL = Distribution Focal Loss**. It is associated with bounding-box localization, specifically YOLO's distribution-based representation of the four box coordinates.

- `train/sem_loss` / `val/sem_loss`: 
	- associated with semantic segmentation rather than the instance masks themselves.
	- Semantic segmentation only distinguishes between separate classes (e.g. belongs to the car  category)
	- Instance segmentation distinguishes between separate objects of the same class. (e.g. this group of pixels belong to car #1 or #2)
	
## 2. Metrics

Because we are performing segmentation, each prediction has both a **bounding box (B)** and a **mask (M)**.

- `metrics/precision(B)` and `metrics/precision(M)`: Asks the question *"Of all the objects YOLO said were objects, how many were actually correct according to the B or M criterion?"*

	Mathematically, 

	$$Precision = \frac{TP}{TP+FP}$$

	where:

	* TP = true positives
	* FP = false positives
	
	Suppose YOLO makes 100 detections:
	
	```text
	90 correct
	10 incorrect
	```
	
	Then:
	
	$$Precision = \frac{90}{100}=0.90$$

	So **90% precision**.

- `metrics/recall(B)` / `metrics/recall(M)`: Asks the opposite question *"Of all the objects that actually exist, how many did YOLO find?"*

  	$$Recall = \frac{TP}{TP+FN}$$
  	
	
	where FN = false negatives.
	
	Suppose there are actually 100 plates in your validation images and YOLO finds 90:
	
	$$Recall = \frac{90}{100}=0.90$$
	
	So **90% recall**.

- `metrics/mAP50(B)` / `metrics/mAP50(M)`

  - **mAP = mean Average Precision.**
  - The `50` means an IoU threshold of **0.50**.  
  - **Intersection over Union, IoU** measures overlap between the predicted and ground-truth regions:

    $$IoU = \frac{\text{intersection}}{\text{union}}$$

    For example:

    ```text
    Ground truth
    ┌───────────────┐
    │               │
    │    plate      │
    │               │
    └───────────────┘

    Prediction
      ┌───────────────┐
      │               │
      │    plate      │
      │               │
      └───────────────┘
    ```

    If they overlap sufficiently, the detection counts as correct.

    At **IoU ≥ 0.50**, it is considered a positive detection for mAP50.

    So:

    ```text
    mAP50(B)
    ```

    asks:

    > How good is the model at detecting the objects using bounding boxes, with IoU ≥ 0.50?

    and:

    ```text
    mAP50(M)
    ```

    asks the equivalent question for **segmentation masks**.

    Higher is better.


- `metrics/mAP50-95(B)` / `metrics/mAP50-95(M)`

  - Generally, this is the **more demanding metric**.
  - Instead of evaluating only at:

    $$IoU=0.50$$

    YOLO evaluates at multiple thresholds:

    $$0.50,\ 0.55,\ 0.60,\ ...,\ 0.95$$

    and averages the resulting AP values.

    Hence:

    > **mAP50-95**

    This is much harder to achieve.

    For example, imagine a detector whose boxes are approximately correct but somewhat loose.

    It might get:

    ```text
    mAP50      = 0.95
    mAP50-95   = 0.70
    ```

    That's not necessarily a bad detector.

    It means:

    > At the relatively forgiving 0.50 IoU threshold, it performs extremely well, but at stricter localization requirements its performance drops.

## 3. TL;DR

Think of the structure of `results.png` as:

```text
                    YOLOv11-seg
                         │
             ┌───────────┴───────────┐
             │                       │
          Training                 Evaluation
             │                       │
          LOSSES                  METRICS
             │                       │
   ┌─────────┼──────────┐       ┌────┼─────────────┐
   │         │          │       │    │             │
  box       seg        cls    precision recall     mAP
   │         │          │       │    │        ┌────┴────┐
  DFL       mask       class    B/M  B/M    mAP50   mAP50-95
```

And the fundamental interpretation is:

| Quantity    | What it measures                       | Better |
| ----------- | -------------------------------------- | ------ |
| `box_loss`  | Bounding-box localization error        | ↓      |
| `seg_loss`  | Mask error                             | ↓      |
| `cls_loss`  | Classification error                   | ↓      |
| `dfl_loss`  | Fine-grained box localization          | ↓      |
| `sem_loss`  | Semantic-segmentation error            | ↓      |
| `precision` | How many detections are correct        | ↑      |
| `recall`    | How many real objects were found       | ↑      |
| `mAP50`     | Detection quality at IoU ≥ 0.50        | ↑      |
| `mAP50-95`  | Detection quality across IoU 0.50–0.95 | ↑      |

## 4. Practical tips when reading `results.png`

- Don't obsess over the **absolute value of the losses**. For example, seeing `box_loss = 0.42` doesn't by itself tell you whether your model is good. Instead, look for patterns such as:

  - Healthy

    ```text
    training loss   ↓↓↓↓↓
    validation loss ↓↓↓↓
    precision       ↑↑↑
    recall          ↑↑↑
    mAP             ↑↑↑
    ```

  - Overfitting

    ```text
    train loss      ↓↓↓↓↓↓↓
    val loss        ↓↓↓ then ↑↑
    train metrics   ↑↑↑
    val metrics     ↑ then ↓
    ```

  - Undertraining

    ```text
    train loss      ↓ ... still decreasing
    val loss        ↓ ... still decreasing
    mAP             ↑ ... still increasing
    ```

    In the last case, stopping at that epoch may simply have been premature.

- Something wrong with the dataset/training setup

  - You can also see situations where losses look reasonable but:

  ```text
  precision ↑
  recall    remains poor
  ```

  which can indicate that the detector is **conservative**—when it predicts an object it's usually right, but it misses many objects.

  - Conversely:

  ```text
  recall ↑↑
  precision ↓
  ```

  means it's finding most objects but producing lots of false detections.




