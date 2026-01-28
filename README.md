# Thai License Plate Recognition (LPR)  Training Workflow

This document outlines the step-by-step process for training the three core models used in the Thai LPR system:

1. **YOLOv8 OBB** 
2. **PP-OCR (Rec)** 
3. **PP-OCR (Det)** 

---

![workflow](asset/mermaid-diagram-2026-01-27-174216.png)

---

## 1. Plate Detection Model (YOLOv8 OBB)

**Goal:** Detect and localize the license plate in the image with rotation (Oriented Bounding Box).

### Prerequisites

* **Labeling Tools:** `X-AnyLabeling` (with `X-AnyLabeling-server` for auto-labeling).
* **Framework:** Ultralytics YOLOv8.

### Workflow Steps

1. **Import Dataset:** Gather raw images containing vehicles/license plates.

2. **Labeling (OBB):**
   
   * Open images in **X-AnyLabeling**.
   * Annotate license plates using the "Rotated Bounding Box" tool.
   * Export labels in **YOLO OBB format** (`.txt` files with `class x y w h angle`).

3. **Dataset Preparation:**
   Organize your files into the standard YOLO structure:
   
   ```text
   datasets/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```
   
   * create **data.yaml** file and set path in that file to locate datasets location, number of class, class name.
   
   * data.yaml example.
     
     ```yaml
     path: /absolute/path/to/datasets
     train: images/train
     val: images/val
     nc: 1
     names: ['license_plate']
     ```

4. **Training:**
   
   * Train the model using the prepared dataset:
     
     ```bash
     yolo obb train model=yolov8s-obb.pt data=path/to/data.yaml batch=64 epochs=100 imgsz=640
     ```

5. **Active Learning (Iterative Loop):**
   
   * Use the newly trained model (`best.pt`) as a pre-trained model in X-AnyLabeling.
   * Run auto-labeling on new images.
   * Manually correct any errors.
   * Re-train the model to improve accuracy.

6. **Export:**
   
   * Once satisfied, export the model to ONNX format for deployment.
     
     ```bash
     yolo export model=best.pt format=onnx opset=12
     ```

---

## 2. Text Recognition Model (PP-OCR Rec)

**Goal:** Read the characters (Thai letters and numbers) from the license plate.

### Prerequisites

* **Tool:** `PPOCRLabel`.
* **Base Model:** Trained YOLOv8 OBB model (from Step 1).
* **Thai dictionary:** contains character for license plate
* **Config Files:** `PP-OCRv5_server_rec.yml` or `PP-OCRv5_mobile_rec.yml`.

### Workflow Steps

1. **Image Pre-processing:**
   
   * Use the **YOLOv8 OBB** model to detect plates.
   * **Crop and Warp:** Crop the detected plate area and apply perspective transform (warp) to make the plate horizontal and rectangular.
   * Save these processed images into a folder.

2. **Labeling:**
   
   * Install and open `PPOCRLabel`.
   * Import the cropped/warped images.
   * Import PP-OCRv5_mobile_rec as pretrained model.
   * Use auto-recognition function to auto-label.
   * Manually correct any errors.
   * **Note:** Ensure Thai characters are labeled correctly.

3. **Export Labels:**
   
   * Export the recognition result in PPOCRLabel.
   * This generates a `rec_gt.txt` file (Format: `image_path \t label`) and Cropped Text Line images.

4. **Training:**
   
   * Configure the training yaml file for the Recognition model.
     
     ```
     PaddleOCR_Dataset/
     ├── train_data/
     │   ├── rec/                      
     │   │   ├── train/                 
     │   │   │   ├── word_001.jpg
     │   │   │   └── ...
     │   │   ├── test/                  
     │   │   │   ├── word_200.jpg
     │   │   │   └── ...
     │   │   └── rec_gt_train.txt       
     │   │   
     │   │
     │   └── det/                       
     │       ├── train/
     │       │   ├── img_001.jpg
     │       │   └── ...
     │       ├── test/
     │       │   ├── img_100.jpg
     │       │   └── ...
     │       └── det_gt_train.txt
     │       
     │
     └── dicts/
     └── th_dict.txt               
     ```
     
     * set path for pretrained model, dictionary, Train data directory, Eval data directory, epoch, etc
     * Train the model using the cropped text line images and `rec_gt.txt`.

5. **Refinement Loop:**
   
   * Use the new Rec model to predict/pre-label new data.
   * Correct the labels in PPOCRLabel.
   * Re-train until the accuracy is satisfactory.

6. **Export Model:**
    *Export to Inference Model:*
    Convert the training checkpoints to an inference model.
   
   ```bash
   python tools/export_model.py -c configs/rec/PP-OCRv5/ch_PP-OCRv5_rec.yml -o Global.pretrained_model=./output/rec/best_accuracy Global.save_inference_dir=./inference/rec/
   ```
   
    *Export Inference model to ONNX format*.
   
   ```bash
   paddle2onnx --model_dir <PATH_TO_INFERENCE_MODEL_FOLDER> \
                 --model_filename inference.pdmodel \
                 --params_filename inference.pdiparams \
                 --save_file <OUTPUT_PATH/model.onnx> \
                 --opset_version 11 \
                 --enable_onnx_checker True
   ```

---

## 3. Text Detection Model (PP-OCR Det)

**Goal:** Locate the text regions (lines of text) within the cropped license plate image.

### Prerequisites

* **Tool:** `PPOCRLabel`.
* **Config Files:** `PP-OCRv5_server_det.yml` or `PP-OCRv5_mobile_det.yml`.

### Workflow Steps

1. **Data Preparation:**
   
   * Use the same cropped license plate images from the Recognition phase.

2. **Labeling:**
   
   * Open `PPOCRLabel`.
   * Draw bounding boxes around each text line/word on the plate (can also use the Label.txt file from Recognition model).
   * Export the result to generate `label.txt` (Det format).

3. **Training Configuration:**
   
   * Select the architecture: **Server** (Accuracy focus) or **Mobile** (Speed focus).
   * Edit the config file (`PP-OCRv5_server_det.yml or PP-OCRv5_mobile_det.yml`) to point to your dataset path.

4. **Training:**
   
   * Run the training script provided by PaddleOCR.
   * Monitor the loss and hmean metrics.

5. **Validation:**
   
   * Test the Det model to ensure it correctly separates lines (e.g., Province name vs. Plate number).

6. **Export Model:**
   
    *Export to Inference Model:*
   
        Convert the training checkpoints to an inference model.
   
   ```bash
   python tools/export_model.py -c configs/det/PP-OCRv5/ch_PP-OCRv5_det.yml -o Global.pretrained_model=./output/det/best_accuracy Global.save_inference_dir=./inference/det/
   ```
   
    *Export Inference model to ONNX format*.
   
   ```bash
   paddle2onnx --model_dir <PATH_TO_INFERENCE_MODEL_FOLDER> \
                   --model_filename inference.pdmodel \
                   --params_filename inference.pdiparams \
                   --save_file <OUTPUT_PATH/model.onnx> \
                   --opset_version 11 \
                   --enable_onnx_checker True
   ```

---

### Summary of Models

| Model Type           | Algorithm  | Input             | Output        | Function                          |
|:-------------------- |:---------- |:----------------- |:------------- |:--------------------------------- |
| **LPR Detection**    | YOLOv8 OBB | Full Image        | Rotated Box   | Finds the plate on the car.       |
| **Text Detection**   | PP-OCR Det | Cropped Plate     | Text Boxes    | Finds where text is on the plate. |
| **Text Recognition** | PP-OCR Rec | Cropped Text Line | String (Text) | Reads the actual characters.      |

```mermaid
graph TD
    %% Define Styles
    classDef data fill:#bbdefb,stroke:#0d47a1,stroke-width:2px,color:#000
    classDef tool fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000
    classDef action fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000
    classDef model fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef decision fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000

    %% === PHASE 1: YOLOv8 OBB ===
    subgraph Phase1 ["🎯 Phase 1: License Plate Detection (YOLOv8 OBB)"]
        direction TB

        RawData["📁 Raw Dataset<br/>(Original Images)"]:::data

        subgraph LabelSetup1 ["Labeling Environment"]
            XAny["🏷️ X-AnyLabeling<br/>(Annotation Tool)"]:::tool
            Server["🖥️ Ubuntu Server<br/>(SAM 3 Model)"]:::tool
            Server <-.->|Auto-labeling Support| XAny
        end

        RawData -->|Import| XAny
        XAny -->|Export Annotations| OBB_Txt["📄 OBB Labels<br/>(.txt format)"]:::data
        OBB_Txt -->|Training Data| TrainOBB["⚙️ Train YOLOv8-OBB<br/>(Oriented Bounding Box)"]:::action
        TrainOBB --> ModelOBB["🤖 YOLOv8-OBB Model<br/>(.pt checkpoint)"]:::model

        %% Iterative Loop
        ModelOBB -.->|Load for Re-labeling| XAny

        %% Export Pipeline
        ModelOBB -->|Convert| ExportONNX1["📦 Export to ONNX<br/>(Intermediate format)"]:::action
        ExportONNX1 -->|Compile| ExportBmodel1["🔧 Export to BModel<br/>(Deployment format)"]:::action
        ExportBmodel1 --> Deploy1["✅ Deployed OBB Model"]:::model
    end

    %% === PHASE 2: PP-OCR Recognition ===
    subgraph Phase2 ["🔤 Phase 2: Text Recognition (PP-OCR Rec)"]
        direction TB

        ModelOBB -->|Crop & Warp Plates| CroppedImg["🖼️ Cropped Plate Images<br/>(Aligned & Normalized)"]:::data

        CroppedImg -->|Load| PPOCR["🏷️ PPOCRLabel<br/>(Text Annotation)"]:::tool

        subgraph QualityControl ["Quality Control"]
            direction LR
            CheckResult{"✓ OCR Quality<br/>Check"}:::decision
            FolderErr["❌ Error Folder<br/>(Misread/Unclear)"]:::data
            FolderOK["✅ Valid Folder<br/>(Correct Labels)"]:::data

            CheckResult -->|Failed| FolderErr
            CheckResult -->|Passed| FolderOK
            FolderErr -.->|Priority Re-labeling| PPOCR
        end

        PPOCR --> CheckResult
        FolderOK -->|Export| RecGT["📄 rec_gt.txt<br/>(image_path\ttext_label)"]:::data
        RecGT -->|Training Data| TrainRec["⚙️ Train PP-OCR Rec<br/>(Recognition Model)"]:::action
        TrainRec --> ModelRec["🤖 PP-OCR Rec Model<br/>(.pdparams)"]:::model

        %% Iterative Loop
        ModelRec -.->|Load for Re-labeling| PPOCR

        %% Export Pipeline
        ModelRec -->|Convert| ExportONNX2["📦 Export to ONNX"]:::action
        ExportONNX2 -->|Compile| ExportBmodel2["🔧 Export to BModel"]:::action
        ExportBmodel2 --> Deploy2["✅ Deployed Rec Model"]:::model
    end

    %% === PHASE 3: PP-OCR Detection ===
    subgraph Phase3 ["🔍 Phase 3: Character Detection (PP-OCR Det)"]
        direction TB

        CroppedImg -.->|Reuse Images| PPOCRDet["🏷️ PPOCRLabel<br/>(Character Box Annotation)"]:::tool

        PPOCRDet -->|Label Character Boxes| DetGT["📄 label.txt<br/>(Quad-point coordinates)"]:::data
        DetGT -->|Training Data| TrainDet["⚙️ Train PP-OCR Det<br/>(Detection Model)"]:::action
        TrainDet --> ModelDet["🤖 PP-OCR Det Model<br/>(.pdparams)"]:::model

        %% Iterative Loop
        ModelDet -.->|Load for Re-labeling| PPOCRDet

        %% Export Pipeline
        ModelDet -->|Convert| ExportONNX3["📦 Export to ONNX"]:::action
        ExportONNX3 -->|Compile| ExportBmodel3["🔧 Export to BModel"]:::action
        ExportBmodel3 --> Deploy3["✅ Deployed Det Model"]:::model
    end

    %% === FINAL INTEGRATION ===
    Deploy1 & Deploy2 & Deploy3 --> FinalPipeline["🎊 Complete OCR Pipeline<br/>(Detection → Recognition → Character Detection)"]:::model

    %% Add legend
    subgraph Legend ["Legend"]
        direction LR
        L1["📁 Data"]:::data
        L2["🏷️ Tool"]:::tool
        L3["⚙️ Action"]:::action
        L4["🤖 Model"]:::model
        L5["✓ Decision"]:::decision
    end
```