# AutoML P&ID Equipment Detector (POC)
## Technologies Used:

* Google Cloud Console (for the UI method) or 
* Google Cloud SDK (gcloud) 
 installed and authenticated on a local machine or Cloud Shell (for the command-line method
* Gemini AI

 
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
3. This data was imported into Vertex AI for further analysis.  Our model underwent training and is visibly in the Vertex AI Model Registry. For this project, our model was named pippingandinstrumentaldiagrams and associated with a dataset ID.
4. To begin training our model for object detectiom we ran 4 versions, altering the average precision, precision, and recall.
5. Next we selected the model with the best training results of Average Precision: 72.5%, Precision: 95.7%, Recall: 71%.
6. This models training capacity consisted of 500 images, 400 training images, 49 validation images, 51 test images on 200 node hours.
7. AutoML image object detection model successfully finished training in 7 hours and 54 minutes, using 67.3 out of a 200 node-hour budget to process the items that were randomly split 80/10/10 for training, validation, and testing.
8. In order to test the model, we deployed it to a Vertex AI endpoint for online predictions. The Vertex AI API was also enabled in your Google Cloud project.  
9. Refined the symbol detection script to generate the full structured metadata output for each image into a table.
10. Refined the symbol detection script to generate an image with detected symbols surrounded by bounding box, symbol name and confidence.
11. Refined the symbol detection script to generate a bar graph to display symbol name and symbol quantity by desired confidence level.
12. Python developer began the process of integrating our deployed model with a web application and chatbot developed by the AutoML trainer.
13.  To deploy the model we used a combination of concepts.  We configured the Endpoint by "Deploying to a new endpoint"  and created an Endpoint Name: for the purpose of the project we chose hello_automl_image.
14. Ensured the region matches the model's region (us-central1). and began configuring the Deployed Model Settings which included: Traffic Split: 100%; Machine Type: AutoML often managed the machine type selection for us.
15. Configured the scaling. In initial testing, setting Minimum compute nodes to 1 was the most cost-effective. Next we ready to deploy the model.
16. Once the deployment process began. Vertex AI provisioned resources and deployed the model container.
17. Next we used the Google Cloud SDK (Command-Line Approach). This method was ideal for automation and integration into scripts.
18.  Next we took the endpoint resource itself. Run the following command in our terminal (Cloud Shell or a locally configured SDK):
Bash
gcloud ai endpoints create \
  --project="PROJECT NAME" \
  --region="REGION" \
  --display-name="hello_automl_image"
19. The endpoint ID can be found in the model registry, or by doing the above command.
20. Next the Python Developer deployed the Model to the Endpoint which was the final step that links the trained model to the live endpoint.
21. The command requires your model ID and the endpoint ID created in step 18.
Bash
gcloud ai models deploy <MODEL_ID> \
  --project="PROJECT NAME" \
  --region="REGION" \
  --endpoint=<ENDPOINT_ID> \
  --display-name="displayname \
  --traffic-split="0=100"
22. The ultimate verification was running our Python script using the final Endpoint ID (19 digit number). We successfully received predictions and generated plots confirms that the model was deployed correctly and is actively serving requests.

## Results
Deployed AutoML model. A prototype performing inference and full metadata extraction for target symbols. Evaluation report complete. Bonus chatbot functional

Protype  script capable of getting detections and attempting OCR-based tag linking. AutoML metrics gained.

## Key Findings
A script's execution environment (including installed libraries like pandas, and the system's PATH variable) is as crucial as the code.

* _Troubleshooting showed how using virtual environments (venv) is essential for managing dependencies locally. We discovered that code that works perfectly in the Cloud Shell can fail on a local machine if the necessary tools aren't installed or accessible. We encountered gcloud: The term 'gcloud' is not recognized and pip: The term 'pip' is not recognized, which were purely environmental setup issues._

TypeError: string indices must be integers deep within the Google authentication library but was caused by a temporary glitch in how the Cloud Shell environment was retrieving its own identity from Google's metadata server._

* _A simple restart of the Cloud Shell session resolved it completely. When working with cloud services, authentication is happening constantly. Internal errors would occur and one of the simplest solutions were to restart, re-authenticate with gcloud auth application-default login._
  
Improved our code based on initial results with iterative approach by continuously refining it based on new requirements and viewing the output.
* _1. Displayed a basic table with tabulate. 2. Changed output to a bar chart with pandas and matplotlib. 3. Saved the plot to a file when it wouldn't display interactively. 4. Refined the visual style: making bars wider, changed the color pattern, and settling on "Google blue bars." 5. Refined the data presentation: adding total counts, and then changing the sorting order multiple times (by count, alphabetically, and finally numerically)._

Building a significant amount of "scaffolding" to transform raw data into different formats.
  
* _The P&ID AutoML model's output was only structured data (lists of names, confidences, and bounding boxes). Initial results were outputed as plots and later into the specific list of dictionaries that our chatbot could understand. Identified data gaps, like the tag_id, which the model doesn't provide and requires another process (like OCR) to fill. The real work was in the "last mile" integration—building the application, translating the model's raw intelligence into a human-usable format was key._
  
Moving the logic from a simple command-line script (detection2.py) to a Flask web application (app.py) highlighted major differences in how applications work.

* _Adapting from a CLI Script to a Web App Involves Fundamental Shifts. We went from a hardcoded GCS_IMAGE_URI to handling dynamic user input via an HTML form that uploads a file._

Rethinking command-line requests

* _The application went from print() statements and saving PNGs directly (plt.savefig) to returning HTML templates and JSON data from API-like routes (/get_chat_response). The plt.show() not working was a clear example of how command-line behavior doesn't translate directly to a web server context. In the web app, we had to manage the state of the application's data (EQUIPMENT_DATA) using a global variable so that it would persist between the image processing step and subsequent chat queries. Deploying a model isn't just about the model itself, but about building a new application paradigm around it, whether it's a simple script or a complex web app._

## Visualizations
