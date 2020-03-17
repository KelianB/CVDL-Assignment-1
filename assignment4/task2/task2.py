import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    int_min = np.maximum(prediction_box[:2], gt_box[:2])
    int_max = np.minimum(prediction_box[2:], gt_box[2:])
    int_lens = int_max - int_min
    if np.any(int_lens < 0):
        return 0
    intersection = np.prod(int_lens)
    # Compute union
    area_prediction = np.prod(prediction_box[2:] - prediction_box[:2])
    area_gt = np.prod(gt_box[2:] - gt_box[:2])
    union = area_prediction + area_gt - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """

    if num_tp + num_fn == 0:
        return 0
    return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    num_predictions = prediction_boxes.shape[0]
    num_gt = gt_boxes.shape[0]

    if num_predictions == 0 or num_gt == 0:
        return np.array([]), np.array([])

    # Find all possible matches with a IoU >= iou threshold
    # The More elegant way to calculate the iou matrix,
    # but does not work for some reason for some input permutations
    #iou_matrix = np.fromfunction(np.vectorize(lambda pred, gt:
    #                             calculate_iou(prediction_boxes[int(pred)], gt_boxes[int(gt)])), shape=(num_predictions, num_gt))
    iou_matrix = np.zeros(shape=(num_predictions, num_gt))
    for i in range(num_predictions):
        for j in range(num_gt):
            iou_matrix[i, j] = calculate_iou(prediction_boxes[i], gt_boxes[j])

    # Sort all matches on IoU in descending order
    sorted_iou_index_matrix = iou_matrix.argsort(axis=1)

    # Find all matches with the highest IoU threshold
    best_gt_match_index = sorted_iou_index_matrix[:, num_gt - 1]
    best_matches_ious = iou_matrix[np.arange(num_predictions), best_gt_match_index]
    matched_predictions_index = np.where(best_matches_ious > iou_threshold)[0]
    matched_gt_index = best_gt_match_index[matched_predictions_index]

    # remove predictions that matched the same ground truth box
    unique_matched_gt_index, indices, counts = np.unique(matched_gt_index, return_index=True, return_counts=True)
    unique_matched_prediction_index = matched_predictions_index[indices]
    for i, c in filter(lambda x: x[1] > 1, enumerate(iter(counts))):
        duplicates = np.where(matched_gt_index == unique_matched_gt_index[i])[0]
        assert(len(duplicates) == c)
        best_box = best_matches_ious[duplicates].argsort()[len(duplicates) - 1]
        unique_matched_prediction_index[i] = duplicates[best_box]

    return prediction_boxes[unique_matched_prediction_index], gt_boxes[unique_matched_gt_index]


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    correct_predictions, _ = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    tp = len(correct_predictions)
    fp = len(prediction_boxes) - tp
    fn = len(gt_boxes) - tp

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    dicts = map(lambda x: calculate_individual_image_result(x[0], x[1], iou_threshold), zip(iter(all_prediction_boxes), iter(all_gt_boxes)))
    x = list(dicts)
    dicts = iter(x)
    results = list(map(sum, zip(*map(lambda x: (x['true_pos'], x['false_pos'], x['false_neg']), dicts))))
    tp = results[0]
    fp = results[1]
    fn = results[2]
    return (calculate_precision(tp, fp, fn), calculate_recall(tp, fp, fn))


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = np.zeros(shape=len(confidence_thresholds))
    recalls = np.zeros(shape=len(confidence_thresholds))
    for i, conf in enumerate(confidence_thresholds):
        scores_conf_iter = map(lambda x: np.where(x > conf), iter(confidence_scores))
        pred_boxes_filtered = list(map(lambda x: x[0][x[1]], zip(iter(all_prediction_boxes), scores_conf_iter)))
        precision, recall = calculate_precision_recall_all_images(pred_boxes_filtered, all_gt_boxes, iou_threshold)
        precisions[i] = precision
        recalls[i] = recall
    return precisions, recalls


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    precision_sum = 0
    for level in recall_levels:
        precisions_filtered = precisions[np.where(recalls >= level)]
        precision_sum += np.max(precisions_filtered) if len(precisions_filtered) > 0 else 0
    average_precision = precision_sum / len(recall_levels)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
