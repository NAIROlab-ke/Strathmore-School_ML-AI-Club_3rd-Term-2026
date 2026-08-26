<div align="center">
<h1>Activities for Term 3, 2026</h1>
<img width="1136" height="574" alt="nairolab-github-banner-cropped" src="https://github.com/NAIROlab-ke/Strathmore-School_ML-AI-Club_3rd-Term-2026/blob/main/nairolab-github-banner-cropped.png" />
</div>

## **Theme: Exploring AI Computer Vision**

### **From Object Detection to Modern Computer Vision**

* Term 2 delivered a **pragmatic jump-start into AI Computer Vision** through a highly practical problem: *detecting and locating multiple object classes in real-world images and video*. This provided a foundation for applications ranging from security and sports perception to object counting and automated inspection. The goal was to:
  * **Learn the core concepts behind Deep Learning for Computer Vision:** dataset collection and evaluation, annotation, preprocessing and augmentation, training and validation, transfer learning, hyperparameters, model performance, and compute resources.
  * **Establish a generic CV workflow:** using **Roboflow, Ultralytics YOLO and Google Colab** to move from dataset to trained model and inference.

* **Term 3 builds on this foundation by expanding our view of what AI can do with visual information.** Having established the basic workflow, we now explore a broader spectrum of Computer Vision tasks: **classification, semantic and instance segmentation, keypoint detection and OCR**, before moving into more advanced **vision-language and multimodal AI**. Models such as **CLIP/SigLIP, Grounding DINO and SAM** allow us to move beyond predefined object classes toward **searching by meaning, locating objects described in natural language, and segmenting arbitrary objects**. The goal is not simply to learn more models, but to understand **the different questions we can ask of visual data—and the different AI techniques used to answer them.**

* **TL;DR**:
  * **Classification** → What is this?
  * **Object Detection (Covered in Term 2)** → *What objects are present, and where are they?*
  * **Segmentation** → Which pixels belong to it?
  * **Keypoints** → Where are its important points?
  * **OCR** → What does it say?
  * **CLIP / SigLIP** → What does this image mean relative to language?
  * **Grounding DINO** → Where is the thing described by this text?
  * **SAM** → Can we precisely segment it?
  * **VLMs** → Can we actually ask questions about the image?

## 1. Image Classification, Sep 3rd

*Question: “What is in this image?”*

* Image classification
* Classes vs. multi-label classification
* Transfer learning
* CNNs → Vision Transformers (conceptual introduction)
* Practical exercise: train/classify a custom dataset

## 2. Image Segmentation, Sep 10th

*Question: “Which pixels belong to what?”*

* **Semantic segmentation** — every pixel gets a class
* **Instance segmentation** — individual objects are separated
* Detection vs. segmentation
* Practical applications: roads, vegetation, people, medical imagery, etc.

## 3. Keypoint / Pose Detection, Sep 17th

*Question: “Where are the important points?”*

* Keypoints
* Human pose estimation
* Hand/face landmarks
* Object keypoints
* Applications: sports analysis, gesture recognition, robotics

## 4. OCR & Document Vision, Sep 24th

*Question: “What does the image say?”*

* Optical Character Recognition (OCR)
* Text detection vs. text recognition
* Reading signs, labels and documents
* Practical OCR pipeline
* Introduction to vision-language models for document understanding

## 5. Computer Vision with Raspberry Pi Cameras, Oct 1st

**A hardware-focused session/lab: Camera → image → CV model → result**

* Standard Raspberry Pi Camera
* Wide-angle camera
* HQ Camera
  * 6 mm vs. 16 mm lenses
  * Field of view
  * Working distance
  * Perspective
* Raspberry Pi AI Camera

<!-- This could culminate in a small experiment where students use different lenses/cameras and observe how **field of view, resolution and distance affect detection**. -->

## 6. Multimodal AI, Oct 8th/15th

We make the transition from *computer vision models* to *models that reason about images and language*.

**Image–Text Embeddings**

* CLIP
* SigLIP
* Image embeddings vs. text embeddings
* Image ↔ text similarity
* Semantic image search

**Open-Vocabulary Detection**

* Grounding DINO
* Text-guided object detection
* “Find the objects described by this text”

**Promptable Segmentation**

* Segment Anything / SAM
* Combining detection and segmentation
* Grounding DINO → SAM pipeline

**Vision-Language Models**

* Image + text → natural-language understanding
* Visual question answering
* Image-based assistants
* Introduction to VLMs

<!--
Comments:

**Don't teach CLIP/SigLIP, Grounding DINO and SAM as three unrelated models.** Make them a single *multimodal vision* story. For example:

**“Find the thing I describe.”**

`Text prompt → Grounding DINO → bounding box → SAM → precise mask`

and then:

**“Find images that mean the same thing.”**

`Image → SigLIP embedding ←→ Text → similarity`

That is likely to be much more memorable for high-school students than a catalog of model names. It also builds very naturally on the YOLO knowledge they already acquired in Term 2.
-->
