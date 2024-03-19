import csv
import logging
import os
import random
import cv2

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()



def plot_segmentation_images_separate(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)      # no mean and std in dataset, replaced with imagenet data, lei, 20240206
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = os.path.split(image_path)[1]
        path_save_all = os.path.join(savefolder, 'all_plots')
        path_save_two_cols = os.path.join(savefolder, 'two_cols')
        path_save_three_cols = os.path.join(savefolder, 'three_cols')
        if not os.path.exists(path_save_all):
            os.mkdir(path_save_all)
        if not os.path.exists(path_save_two_cols):
            os.mkdir(path_save_two_cols)
        if not os.path.exists(path_save_three_cols):
            os.mkdir(path_save_three_cols)

        img_orig = cv2.cvtColor(image.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
        # min_s = np.min(segmentation)
        # max_s = np.max(segmentation)
        # segmentation_color = cv2.applyColorMap(((segmentation - min_s) / (max_s - min_s) * 255).astype(np.uint8),
        #                                        cv2.COLORMAP_JET)
        segmentation_color = cv2.applyColorMap((segmentation * 255).astype(np.uint8),
                                               cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_orig.png'), img_orig)
        cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_seg.png'), segmentation_color)
        if masks_provided:
            img_orig_edge = np.zeros_like(img_orig)
            np.copyto(img_orig_edge, img_orig)
            mask_gray = cv2.cvtColor((mask.astype(np.uint8) * 255).transpose((1, 2, 0)), cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(mask_gray, 0, cv2.CHAIN_APPROX_NONE)
            img_orig_edge = cv2.drawContours(img_orig_edge, contours, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_orig_with_contours.png'), img_orig_edge)
            cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_mask.png'), mask_gray)

        # img_orig = cv2.copyMakeBorder(img_orig, 5, 5, 5, 5,
        #                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # segmentation_color = cv2.copyMakeBorder(segmentation_color, 5, 5, 5, 5,
        #                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))

        im_strip = np.ones((img_orig.shape[0], 10, 3), dtype=np.uint8) * 255
        if masks_provided:
            # img_orig_edge = cv2.copyMakeBorder(img_orig_edge, 5, 5, 5, 5,
            #                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
            mask_gray_1d = np.expand_dims(mask_gray, axis=2)
            mask_gray_3d = np.concatenate([mask_gray_1d, mask_gray_1d, mask_gray_1d], axis=2)
            # mask_gray_3d = cv2.copyMakeBorder(mask_gray_3d, 5, 5, 5, 5,
            #                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
            im_combine = np.concatenate([img_orig_edge, im_strip, segmentation_color], axis=1)
            cv2.imwrite(os.path.join(savefolder, 'two_cols', savename[: -4] + '.png'), im_combine)
            im_combine = np.concatenate([img_orig, im_strip, mask_gray_3d, im_strip, segmentation_color], axis=1)
            cv2.imwrite(os.path.join(savefolder, 'three_cols', savename[: -4] + '.png'), im_combine)
        else:
            im_combine = np.concatenate([img_orig, segmentation_color], axis=1)
            cv2.imwrite(os.path.join(savefolder, 'two_cols', savename[: -4] + '.png'), im_combine)


def plot_segmentation_images_separate_compare(
    savefolder,
    image_paths,
    segmentations,
    segmentations_ori,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation, segmentation_ori in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations, segmentations_ori),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)      # no mean and std in dataset, replaced with imagenet data, lei, 20240206
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        # savename = os.path.split(image_path)[1]
        savename = '_'.join(image_path.split('/')[-2:])
        path_save_all = os.path.join(savefolder, 'all_plots')
        path_save_two_cols = os.path.join(savefolder, 'two_cols')
        path_save_three_cols = os.path.join(savefolder, 'three_cols')
        path_save_compare_cols = os.path.join(savefolder, 'compare_cols')
        if not os.path.exists(path_save_all):
            os.mkdir(path_save_all)
        if not os.path.exists(path_save_two_cols):
            os.mkdir(path_save_two_cols)
        if not os.path.exists(path_save_three_cols):
            os.mkdir(path_save_three_cols)
        if not os.path.exists(path_save_compare_cols):
            os.mkdir(path_save_compare_cols)
            
        img_orig = cv2.cvtColor(image.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
        # min_s = np.min(segmentation)
        # max_s = np.max(segmentation)
        # segmentation_color = cv2.applyColorMap(((segmentation - min_s) / (max_s - min_s) * 255).astype(np.uint8),
        #                                        cv2.COLORMAP_JET)
        segmentation_color = cv2.applyColorMap((segmentation * 255).astype(np.uint8),
                                               cv2.COLORMAP_JET)
        segmentation_color_ori = cv2.applyColorMap((segmentation_ori * 255).astype(np.uint8),
                                               cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_orig.png'), img_orig)
        cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_seg.png'), segmentation_color)
        cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_ori_seg.png'), segmentation_color_ori)
        if masks_provided:
            img_orig_edge = np.zeros_like(img_orig)
            np.copyto(img_orig_edge, img_orig)
            mask_gray = cv2.cvtColor((mask.astype(np.uint8) * 255).transpose((1, 2, 0)), cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(mask_gray, 0, cv2.CHAIN_APPROX_NONE)
            img_orig_edge = cv2.drawContours(img_orig_edge, contours, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_orig_with_contours.png'), img_orig_edge)
            cv2.imwrite(os.path.join(path_save_all, savename[: -4] + '_mask.png'), mask_gray)

        # img_orig = cv2.copyMakeBorder(img_orig, 5, 5, 5, 5,
        #                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # segmentation_color = cv2.copyMakeBorder(segmentation_color, 5, 5, 5, 5,
        #                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))

        im_strip = np.ones((img_orig.shape[0], 10, 3), dtype=np.uint8) * 255
        if masks_provided:
            # img_orig_edge = cv2.copyMakeBorder(img_orig_edge, 5, 5, 5, 5,
            #                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
            mask_gray_1d = np.expand_dims(mask_gray, axis=2)
            mask_gray_3d = np.concatenate([mask_gray_1d, mask_gray_1d, mask_gray_1d], axis=2)
            # mask_gray_3d = cv2.copyMakeBorder(mask_gray_3d, 5, 5, 5, 5,
            #                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
            im_combine = np.concatenate([img_orig_edge, im_strip, segmentation_color], axis=1)
            cv2.imwrite(os.path.join(savefolder, 'two_cols', savename[: -4] + '.png'), im_combine)
            im_combine = np.concatenate([img_orig, im_strip, mask_gray_3d, im_strip, segmentation_color], axis=1)
            cv2.imwrite(os.path.join(savefolder, 'three_cols', savename[: -4] + '.png'), im_combine)
            im_combine = np.concatenate([img_orig_edge, im_strip, segmentation_color_ori, im_strip, segmentation_color], axis=1)
            cv2.imwrite(os.path.join(savefolder, 'compare_cols', savename[: -4] + '.png'), im_combine)
        else:
            im_combine = np.concatenate([img_orig, segmentation_color], axis=1)
            cv2.imwrite(os.path.join(savefolder, 'two_cols', savename[: -4] + '.png'), im_combine)


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
    name = "results",
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, f"{name}.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
