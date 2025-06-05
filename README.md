# AutoML P&ID Equipment Detector (POC)

## Table of Contents
- [Project Overview](https://github.com/JasmineH12/Sprinternship/blob/main/README.md)
- [Objective](link)
- [Goals](link)
- [Methodology](link)
- [Results](link)
- [Key Findings](link)
- [Visualizations](link)
- [Future Work](link)
- [Individual Contributions](link)

## Objective
A team of students developed a Proof-of-Concept using Google Cloud to identify specific equipment symbols from Piping and Instrumentation Diagrams (P&IDs). A bonus includes a simple chatbot to query the detected equipment data in two weeks.

## Goals
1. reformat the annotations given in the dataset for the AutoML expected format for downstream tasks.
2. Upload anotated images for a viable initial training dataset.
3. Analyze the AutoML-provided evaluation metrics (mAP, precision/recall per label).
4. Develop a Python script to Send a new (test) P&ID image to the deployed model endpoint to get symbol detections (class, bounding box, confidence).
5. Once reasonably satisfied with a model iteration, deploy it to a Vertex AI Endpoint for online predictions
6. Generate the full structured metadata output (symbol type, tag ID, position, confidence, structured snippet) for each detected target symbol.
7. Document the final model's performance and the accuracy of the end-to-end metadata extraction
8.  Develop a simple chatbot interface (CLI or basic web UI) to query the extracted equipment metadata from a processed P&ID
9.  Finalize evaluation report, and prepare project demo/presentation.

## Methodology
1. The data engineer and technical assistant used a python code, located in the [here] to reformat numpy files from the digitized dataset.
2. Once the data was reformated and values were normalized for each bounding box, the file was then uploaded to the bucket in Google Cloud Storage.
3. This data was imported into Vertex AI for further analysis.
4. 

## Results
Deployed AutoML model. A prototype performing inference and full metadata extraction for target symbols. Evaluation report complete. Bonus chatbot functional


  
