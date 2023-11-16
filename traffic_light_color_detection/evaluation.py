from shapely.geometry import box
from shapely.ops import unary_union


# convert the cvs data of ground truth and predictions to a dictionart
def csv_to_dict(df):
    # Format csv data to dictonary
    dict_data = {}
    for idx, row in df.iterrows():
        filename = row['filename']
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        label = row['color']

        if filename not in dict_data:
            dict_data[filename] = []

        dict_data[filename].append({'bbox': bbox, 'label': label})
    return dict_data

def calculate_iou(gt_bbox, pred_bbox):
    # Creating shapely objects for ground truth and predicted bounding boxes
    gt_box = box(*gt_bbox)
    pred_box = box(*pred_bbox)
    
    # Calculating intersection and union
    intersection = gt_box.intersection(pred_box).area
    union = unary_union([gt_box, pred_box]).area
    
    # Calculating IoU
    iou = intersection / union
    return iou


# pass the dictionary of ground truth and prediction to this function to get the accuracy, precision, recall and F1 score
def evaluate_result(ground_truth, pred):
    iou_threshold = 0.5

    predictions = {}

    # Initializing counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_instances = 0

    # Evaluating predictions
    for filename, pred in ground_truth.items():
        if filename in ground_truth:
            gt_list = ground_truth[filename]

    #         total_instances += 1
            total_instances += len(gt_list)

            for pred in pred:
                pred_bbox = pred['bbox']

                # Finding the ground truth bounding box with highest IoU
                max_iou = 0
                match_found = False

                for gt in gt_list:
                    gt_bbox = gt['bbox']
                    iou = calculate_iou(gt_bbox, pred_bbox)

                    if iou > max_iou:
                        max_iou = iou

                if max_iou >= iou_threshold:
                    true_positives += 1
                else:
                    false_positives += 1


    false_negatives = total_instances - true_positives

    # Accuracy
    true_negatives = total_instances - (true_positives + false_positives + false_negatives)
    accuracy = (true_positives + true_negatives)/total_instances

    # Calculating precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return(accuracy, precision, recall, f1_score)