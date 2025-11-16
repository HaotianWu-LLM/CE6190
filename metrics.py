import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import roc_auc_score, precision_recall_curve
import cv2


class SegmentationMetrics:
    """Calculate segmentation metrics for binary masks"""

    @staticmethod
    def dice_coefficient(pred, target, smooth=1e-6):
        """Calculate Dice Coefficient"""
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    @staticmethod
    def iou(pred, target, smooth=1e-6):
        """Calculate Intersection over Union (Jaccard Index)"""
        pred = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def precision(pred, target, smooth=1e-6):
        """Calculate Precision"""
        pred = pred.flatten()
        target = target.flatten()
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        return (tp + smooth) / (tp + fp + smooth)

    @staticmethod
    def recall(pred, target, smooth=1e-6):
        """Calculate Recall (Sensitivity)"""
        pred = pred.flatten()
        target = target.flatten()
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        return (tp + smooth) / (tp + fn + smooth)

    @staticmethod
    def hausdorff_distance(pred, target):
        """Calculate Hausdorff Distance"""
        pred_points = np.argwhere(pred > 0)
        target_points = np.argwhere(target > 0)

        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')

        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        return max(hd1, hd2)

    @staticmethod
    def auc_roc(pred_prob, target):
        """Calculate AUC-ROC"""
        pred_prob = pred_prob.flatten()
        target = target.flatten()

        if len(np.unique(target)) < 2:
            return np.nan

        return roc_auc_score(target, pred_prob)

    @staticmethod
    def calculate_all_metrics(pred, target, pred_prob=None):
        """Calculate all metrics"""
        # Ensure binary masks
        pred = (pred > 0.5).astype(np.float32)
        target = (target > 0.5).astype(np.float32)

        metrics = {
            'dice': float(SegmentationMetrics.dice_coefficient(pred, target)),
            'iou': float(SegmentationMetrics.iou(pred, target)),
            'precision': float(SegmentationMetrics.precision(pred, target)),
            'recall': float(SegmentationMetrics.recall(pred, target)),
            'hausdorff': float(SegmentationMetrics.hausdorff_distance(pred, target))
        }

        if pred_prob is not None:
            metrics['auc_roc'] = float(SegmentationMetrics.auc_roc(pred_prob, target))

        return metrics


class MetricsTracker:
    """Track and aggregate metrics across dataset"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics_list = []

    def update(self, pred, target, pred_prob=None):
        """Add new prediction to tracker"""
        metrics = SegmentationMetrics.calculate_all_metrics(pred, target, pred_prob)
        self.metrics_list.append(metrics)

    def get_average_metrics(self):
        """Calculate average metrics"""
        if not self.metrics_list:
            return {}

        avg_metrics = {}
        for key in self.metrics_list[0].keys():
            values = [m[key] for m in self.metrics_list if not np.isnan(m[key])]
            avg_metrics[key] = np.mean(values) if values else np.nan
            avg_metrics[f'{key}_std'] = np.std(values) if values else np.nan

        return avg_metrics

    def print_metrics(self):
        """Print average metrics"""
        avg_metrics = self.get_average_metrics()
        print("\n" + "=" * 50)
        print("Average Metrics:")
        print("=" * 50)
        for key, value in avg_metrics.items():
            if not key.endswith('_std'):
                std_key = f'{key}_std'
                std_value = avg_metrics.get(std_key, 0)
                print(f"{key.upper():15s}: {value:.4f} ± {std_value:.4f}")
        print("=" * 50 + "\n")