import base64
from io import BytesIO
import os
import google.cloud.aiplatform as aiplatform
from google.cloud import storage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
import pandas as pd

# endpoint localization & confidence filter 
PROJECT_ID = "YOUR PROJECT ID"
LOCATION = "us-central1"
ENDPOINT_ID = "YOUR ENDPOINT ID"
GCS_IMAGE_URI = "GCS IMAGE URI PATH"
CONFIDENCE_FILTER = 0.6

def predict_image_object_detection(
    project_id: str,
    location: str,
    endpoint_id: str,
    gcs_image_uri: str
):
    aiplatform.init(project=project_id, location=location)
    storage_client = storage.Client(project=project_id)

    path_parts = gcs_image_uri.replace("gs://", "").split("/", 1)
    if len(path_parts) < 2 or not path_parts[0] or not path_parts[1]:
        print(f"Error: GCS URI '{gcs_image_uri}' is not in 'gs://bucket/object' format.")
        return None, None
    bucket_name, blob_name = path_parts[0], path_parts[1]

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    image_bytes = blob.download_as_bytes()

    encoded_content = base64.b64encode(image_bytes).decode("utf-8")
    instances = [{"content": encoded_content}]

    endpoint_path = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_path)

    prediction_response = endpoint.predict(instances=instances)
    print("Prediction successful!")
    return prediction_response, image_bytes

# assigns the symbol class to a name
def plot_symbol_counts_barchart(prediction_response, confidence_threshold=0.0):
    symbol_names_map = {
        "1": "Gate Valve",
        "2": "Cross Ball Valve",
        "3": "Globe Valve",
        "4": "Valve",
        "5": "Ball Valve",
        "6": "Butterfly Valve",
        "7": "Plug Valve",
        "8": "Diode",
        "9": "Diaphragm Valve",
        "10": "Needle Valve",
        "11": "Closed Gate Valve",
        "12": "Normally Closed Gate Valve",
        "13": "Normally Closed Ball Valve",
        "14": "Control Valve",
        "15": "Rotary Valve",
        "16": "Closed Rotary Valve",
        "17": "Spacer Ring",
        "18": "Closed Spectacle Blind",
        "19": "Open Spectacle Blind",
        "20": "Concentric Reducer",
        "21": "Flanged Connection",
        "22": "Filter",
        "23": "Heat Exchanger",
        "24": "Flow Direction",
        "25": "Circle Valve",
        "26": "GRI-808",
        "27": "R0-10-871",
        "28": "SDL-973",
        "29": "DDL-686",
        "30": "STA",
        "31": "ZL0-946",
        "32": "121-LG-10-190"
    }

    detections = prediction_response.predictions[0]

    display_names_all = detections.get('displayNames', [])
    confidences_all = detections.get('confidences', [])

    filtered_display_names = [
        display_names_all[i]
        for i in range(len(display_names_all))
        if float(confidences_all[i]) >= confidence_threshold
    ]

    if not filtered_display_names:
        print(f"No symbols found meeting confidence threshold >= {confidence_threshold*100:.0f}% to plot counts.")
        return

    total_symbols_detected = len(filtered_display_names)
    symbols_series = pd.Series(filtered_display_names)
    symbol_counts = symbols_series.value_counts()

    if symbol_counts.empty:
        print(f"No symbols counted after filtering (threshold {confidence_threshold*100:.0f}%). Cannot plot bar chart.")
        return

    numeric_index_sorted = pd.to_numeric(symbol_counts.index).sort_values(ascending=True)
    symbol_counts_sorted_by_numeric_key = symbol_counts.reindex(numeric_index_sorted.astype(str))

    new_labels = [symbol_names_map.get(idx, idx) for idx in symbol_counts_sorted_by_numeric_key.index]
    plot_series = pd.Series(data=symbol_counts_sorted_by_numeric_key.values, index=new_labels)
    xlabel_text = 'Symbol Type'

    plt.figure(figsize=(20, 15)) 
    google_blue_hex = '#4285F4'
    plot_series.plot(kind='bar', color=google_blue_hex, edgecolor='black', width=0.8)

    title_text = (f'Detected P&ID Symbol Counts (Total: {total_symbols_detected})\n'
                  f'(Confidence >= {confidence_threshold*100:.0f}%)')
    plt.title(title_text)

    plt.xlabel(xlabel_text)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha="right") 
    if not plot_series.empty:
        plt.yticks(range(0, int(plot_series.max()) + 2,
                         max(1, int(plot_series.max()) // 10 or 1)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    ax = plt.gca()
    for i, bar_obj in enumerate(ax.patches):
        count = bar_obj.get_height()
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2.,
                count + 0.05,
                str(int(count)),
                ha='center', va='bottom')

    plt.tight_layout()

    plot_filename = "symbol_counts_barchart.png"
    plt.savefig(plot_filename)
    print(f"\nSymbol Count Bar Chart saved. ")
    plt.close()

def plot_object_detection_results(prediction_response, image_bytes, image_uri=None, confidence_threshold=0.5):

    fig = None
    try:
        img = PIL.Image.open(BytesIO(image_bytes))
        fig_width = 12
        fig_height = fig_width * img.height / img.width if img.width > 0 and img.height > 0 else fig_width
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(img)
        ax = plt.gca()

        detections = prediction_response.predictions[0]

        display_names = detections.get('displayNames', [])
        confidences = detections.get('confidences', [])
        bboxes_normalized = detections.get('bboxes', [])

        img_width_px, img_height_px = img.size
        num_plotted_detections = 0

        for i in range(len(display_names)):
            score = float(confidences[i])
            if score >= confidence_threshold:
                num_plotted_detections += 1
                label_id = display_names[i] # This is the numeric ID "1", "2", etc.
            
                try:
                    temp_symbol_names_map = {
                        "1": "Gate Valve", "2": "Cross Ball Valve", "3": "Globe Valve", "4": "Valve",
                        "5": "Ball Valve", "6": "Butterfly Valve", "7": "Plug Valve", "8": "Diode",
                        "9": "Diaphragm Valve", "10": "Needle Valve", "11": "Closed Gate Valve", "12": "Normally Closed Gate Valve",
                        "13": "Normally Closed Ball Valve", "14": "Control Valve", "15": "Rotary Valve", "16": "Closed Rotary Valve",
                        "17": "Spacer Ring", "18": "Closed Spectacle Blind", "19": "Open Spectacle Blind", "20": "Concentric Reducer",
                        "21": "Flanged Connection", "22": "Filter", "23": "Heat Exchanger", "24": "Flow Direction",
                        "25": "Circle Valve", "26": "GRI-808", "27": "R0-10-871", "28": "SDL-973",
                        "29": "DDL-686", "30": "STA", "31": "ZL0-946", "32": "21-LG-10-190"
                    }
                    descriptive_label = temp_symbol_names_map.get(label_id, label_id)
                except NameError: 
                    descriptive_label = label_id
            
                box = bboxes_normalized[i]

                xmin_abs = box[0] * img_width_px
                xmax_abs = box[1] * img_width_px
                ymin_abs = box[2] * img_height_px
                ymax_abs = box[3] * img_height_px

                rect_width = xmax_abs - xmin_abs
                rect_height = ymax_abs - ymin_abs

                rect = patches.Rectangle(
                    (xmin_abs, ymin_abs), rect_width, rect_height,
                    linewidth=1, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    xmin_abs, ymin_abs - 10,
                    f'{descriptive_label}: {score:.2f}', 
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor='red', alpha=0.7, pad=0.5)
                )

        title_str = f"Object Detection Results (Total Plotted: {num_plotted_detections})"
        if image_uri:
            title_str += f"\n({image_uri.split('/')[-1]})"
        title_str += f" (Conf >= {confidence_threshold*100:.0f}%)"
        plt.title(title_str)
        plt.axis('off')

        plot_filename = "detection_results_with_boxes.png"
        plt.savefig(plot_filename)
        print(f"\nImage with Bounding Boxes saved.")

        if num_plotted_detections == 0:
            print(f"No objects plotted with confidence >= {confidence_threshold*100:.0f}%.")

    finally:
        if fig:
            plt.close(fig)


if __name__ == "__main__":

        prediction_result_obj, downloaded_image_bytes = predict_image_object_detection(
            project_id=PROJECT_ID,
            location=LOCATION,
            endpoint_id=ENDPOINT_ID,
            gcs_image_uri=GCS_IMAGE_URI
        )

        if prediction_result_obj:
            plot_symbol_counts_barchart(prediction_result_obj, confidence_threshold=CONFIDENCE_FILTER)

            if downloaded_image_bytes:
                plot_object_detection_results(prediction_result_obj, downloaded_image_bytes, GCS_IMAGE_URI, confidence_threshold=CONFIDENCE_FILTER)
            else:
                print("Image bytes not available, cannot plot bounding boxes on image.")

        elif downloaded_image_bytes:
            print("\nPrediction failed (no result object was returned), but image was downloaded.")

        else:
            print("\nPrediction process failed. Image might not have been downloaded or an early error occurred.")

print("\nScript finished!")
