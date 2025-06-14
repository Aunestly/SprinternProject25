# AutoML P&ID Equipment Detector (POC)

## Table of Contents
- [Project Overview](https://github.com/JasmineH12/Sprinternship/blob/main/README.md)
- [Objective](https://github.com/Aunestly/SprinternProject25?tab=readme-ov-file#objective)
- [Goals](https://github.com/Aunestly/SprinternProject25?tab=readme-ov-file#goals)
- [Methodology](https://github.com/Aunestly/SprinternProject25?tab=readme-ov-file#methodology)
- [Results](https://github.com/Aunestly/SprinternProject25?tab=readme-ov-file#results)
- [Key Findings](link)
- [Visualizations](link)
- [Future Work](link)
- [Individual Contributions](link)

## Objective
A team of students developed a Proof-of-Concept using Google Cloud to identify specific equipment symbols from Piping and Instrumentation Diagrams (P&IDs). A bonus includes a simple chatbot to query the detected equipment data in two weeks.

## Goals
1. Reformat the annotations given in the dataset for the AutoML expected format for downstream tasks.
2. Upload anotated images for a viable initial training dataset.
3. Analyze the AutoML-provided evaluation metrics (mAP, precision/recall per label).
4. Protype  script capable of getting detections and attempting OCR-based tag linking.
6. Once reasonably satisfied with a model iteration, deploy it to a Vertex AI Endpoint for online predictions
7. Send a new (test) P&ID image to the deployed model's endpoint to get symbol detection results (class, bounding box, confidence)
8. Generate the full structured metadata output (symbol type, tag ID, position, confidence, structured snippet) for each detected target symbol.
9. Document the final model's performance and the accuracy of the end-to-end metadata extraction.
10. Develop a simple chatbot interface (CLI/basic web UI) to query extracted equipment metadata from a processed P&ID
11. Finalize evaluation report, and prepare project demo/presentation.

## Methodology
1. The data engineer and technical assistant used a python code, located in [here](https://github.com/JasmineH12/Sprinternship/blob/main/converting_to_ML_readable.py) to reformat numpy files.
2. Once the data was reformated and values were normalized for each bounding box, the file was then uploaded to the bucket in Google Cloud Storage.
3. This data was imported into Vertex AI for further analysis.
4. To begin training our model for object detectiom we ran 4 versions, altering the average precision, precision, and recall.
5. Next we selected the model with the best training results of Average Precision: 72.5%, Precision: 95.7%, Recall: 71%.
6. This models training capacity consisted of 500 images, 400 training images, 49 validation images, 51 test images on 200 node hours.
7. AutoML image object detection model successfully finished training in 7 hours and 54 minutes, using 67.3 out of a 200 node-hour budget to process the items that were randomly split 80/10/10 for training, validation, and testing.
8. In order to test the model, we deployed it to a Vertex AI endpoint for online predictions.
9. Refined the symbol detection script to generate the full structured metadata output for each image into a table.
10. Refined the symbol detection script to generate an image with detected symbols surrounded by bounding box, symbol name and confidence.
11. Refined the symbol detection script to generate a bar graph to display symbol name and symbol quantity by desired confidence level.
12. Python developer Integrated deployed model with a web application and chatbot developed by the AutoML trainer.
13. 

## Results
Deployed AutoML model. A prototype performing inference and full metadata extraction for target symbols. Evaluation report complete. Bonus chatbot functional

Protype  script capable of getting detections and attempting OCR-based tag linking. AutoML metrics gained.


  
