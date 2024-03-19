# ------------------------------------------------------------------
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# ------------------------------------------------------------------
# Modified by SNHIPOW
# ------------------------------------------------------------------
import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch
import tqdm
import random
from torch import nn
import torch.optim as optim

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
import patchcore.projection

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],"visa": ["patchcore.datasets.visa", "VisADataset"]}

@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--epochs", type=int, default=40)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="debug")
@click.option("--log_project", type=str, default="test")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    epochs,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    methods = {key: item for (key, item) in methods}

    os.makedirs(results_path, exist_ok=True)
    project_path = os.path.join(results_path, log_project)
    os.makedirs(project_path, exist_ok=True)
    run_save_path = os.path.join(project_path, log_group)
    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    result_I_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )
        patchcore_save_path = os.path.join(
            run_save_path, "models", dataloaders["training"].name)
        if os.path.exists(patchcore_save_path):
            continue
        os.makedirs(patchcore_save_path, exist_ok=True)
        
        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](device)
            result_txt = f'{log_group}.txt'

            ###
            projection_lst,  jigsawcls_lst, jigsaw_optim_lst = [], [], []
            for i in range(3):
                projection_lst.append(methods["get_projection"](device))    
                jigsawcls_lst.append(methods["get_jigsawcls"](device))     
                jigsaw_optim_lst.append({"criterion":nn.CrossEntropyLoss(reduction='mean'),
                                   "optimizer_p":optim.Adam(params=projection_lst[i].parameters(), lr=1e-4),
                                   "optimizer_j":optim.Adam(params=jigsawcls_lst[i].parameters(), lr=1e-4)})
                
            jigsaw_cls_num = jigsawcls_lst[0]._cls_num()
            
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, projection_lst, device)
            ###
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
                
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)

            for i_epoch in range(epochs): 
                LOGGER.info(
                    "training [{}] ({}/{})...".format(
                        dataloaders["training"].name,dataloader_count + 1,len(list_of_dataloaders),
                        )
                    )
                for model_i, PatchCore in enumerate(PatchCore_list):
                    projection_lst[model_i].train()
                    jigsawcls_lst[model_i].train()
                    ###########################
                    with tqdm.tqdm(
                            dataloaders["jigsaw_training"], desc=f"Epoch:{i_epoch}, jigsaw_training:{model_i+1}/{len(PatchCore_list)}", position=1, leave=False
                        ) as data_iterator:
                        for image in data_iterator:
                            if isinstance(image, dict):
                                image = image["image"].to(torch.float).to(device)
                                img_w = image.size(-1)
                                
                            spatial_perm = None
                            for batch_i in range(image.shape[0]):
                                if random.random() < 0.0001:
                                    spatial_perm_ = np.arange(jigsaw_cls_num**2)[None,:]
                                else:
                                    spatial_perm_ = np.random.permutation(jigsaw_cls_num**2)[None,:]
                                    
                                if spatial_perm is None:
                                    spatial_perm = spatial_perm_
                                else:
                                    spatial_perm = np.concatenate([spatial_perm,spatial_perm_],axis=0)
                                    
                            jigsaw_patch_size = img_w//jigsaw_cls_num
                            border = (imagesize[-1]-jigsaw_patch_size*jigsaw_cls_num)//2
                            for batch_i in range(image.shape[0]):
                                img_patch_list = []
                                for i in range(jigsaw_cls_num):
                                    for j in range(jigsaw_cls_num):
                                        y_offset = border + jigsaw_patch_size * j
                                        x_offset = border + jigsaw_patch_size * i
                                        image_patch = image[batch_i, :, y_offset: y_offset + jigsaw_patch_size, x_offset: x_offset + jigsaw_patch_size]
                                        img_patch_list.append(image_patch)
                                for p_ind, i in enumerate(spatial_perm[batch_i]):
                                    # y = i // jigsaw_cls_num
                                    # x = i % jigsaw_cls_num
                                    y = i % jigsaw_cls_num
                                    x = i // jigsaw_cls_num
                                    y_offset = border + jigsaw_patch_size * y
                                    x_offset = border + jigsaw_patch_size * x
                                    image[batch_i, :, y_offset: y_offset + jigsaw_patch_size, x_offset: x_offset + jigsaw_patch_size] = img_patch_list[p_ind]
                            
                            features = PatchCore._embed(image)
                            features = np.asarray(features)
                            features = torch.from_numpy(features).to(device)
                            features = projection_lst[model_i](features)

                            c_d = features.shape[-1]
                            features = features.reshape(-1, img_w//8, img_w//8, c_d).permute(0,3,1,2)
                        
                            spat_logits = jigsawcls_lst[model_i](features)
                            spat_logits = spat_logits.view(-1,jigsaw_cls_num**2)
                            spat_label = torch.from_numpy(spatial_perm).long().view(-1).to(device)
                            spat_loss = jigsaw_optim_lst[model_i]["criterion"](spat_logits, spat_label)
                            
                            jigsaw_optim_lst[model_i]["optimizer_p"].zero_grad()
                            jigsaw_optim_lst[model_i]["optimizer_j"].zero_grad()
                            spat_loss.backward()
                            jigsaw_optim_lst[model_i]["optimizer_p"].step()
                            jigsaw_optim_lst[model_i]["optimizer_j"].step()
                            # print(spat_loss)
                                    
                        ###########################
                PatchCore_list = methods["get_patchcore"](imagesize, sampler, projection_lst, device)
                for i, PatchCore in enumerate(PatchCore_list):
                    torch.cuda.empty_cache()
                    if PatchCore.backbone.seed is not None:
                        patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                    LOGGER.info(
                        "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                    )
                    PatchCore.fit(dataloaders["training"])

                torch.cuda.empty_cache()
                aggregator = {"scores": [], "segmentations": []}
                for i, PatchCore in enumerate(PatchCore_list):
                    torch.cuda.empty_cache()
                    LOGGER.info(
                        "Embedding test data with models ({}/{})".format(
                            i + 1, len(PatchCore_list)
                        )
                    )
                    scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                        dataloaders["testing"]
                    )
                    aggregator["scores"].append(scores)
                    aggregator["segmentations"].append(segmentations)

                scores = np.array(aggregator["scores"])
                min_scores = scores.min(axis=-1).reshape(-1, 1)
                max_scores = scores.max(axis=-1).reshape(-1, 1)
                scores = (scores - min_scores) / (max_scores - min_scores)
                scores = np.mean(scores, axis=0)

                segmentations = np.array(aggregator["segmentations"])
                min_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .min(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                max_scores = (
                    segmentations.reshape(len(segmentations), -1)
                    .max(axis=-1)
                    .reshape(-1, 1, 1, 1)
                )
                segmentations = (segmentations - min_scores) / (max_scores - min_scores)
                segmentations = np.mean(segmentations, axis=0)

                anomaly_labels = [
                    x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                # (Optional) Plot example images.
                if save_segmentation_images:
                    image_paths = [
                        x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                    ]
                    mask_paths = [
                        x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                    ]

                    def image_transform(image):
                        in_std = np.array(
                            dataloaders["testing"].dataset.transform_std
                        ).reshape(-1, 1, 1)
                        in_mean = np.array(
                            dataloaders["testing"].dataset.transform_mean
                        ).reshape(-1, 1, 1)
                        image = dataloaders["testing"].dataset.transform_img(image)
                        return np.clip(
                            (image.numpy() * in_std + in_mean) * 255, 0, 255
                        ).astype(np.uint8)

                    def mask_transform(mask):
                        return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                    image_save_path = os.path.join(
                        run_save_path, "segmentation_images", dataset_name
                    )
                    os.makedirs(image_save_path, exist_ok=True)
                    patchcore.utils.plot_segmentation_images(
                        image_save_path,
                        image_paths,
                        segmentations,
                        scores,
                        mask_paths,
                        image_transform=image_transform,
                        mask_transform=mask_transform,
                    )

                LOGGER.info("Computing evaluation metrics.")
                auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                    scores, anomaly_labels
                )["auroc"]

                # Compute PRO score & PW Auroc for all images
                pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                    segmentations, masks_gt
                )
                full_pixel_auroc = pixel_scores["auroc"]

                pro_auc = patchcore.metrics.compute_pro(masks_gt,segmentations)      

                if i_epoch == 0:
                    result_collect.append(
                        {
                            "dataset_name": dataset_name,
                            "instance_auroc": auroc,
                            "full_pixel_auroc": full_pixel_auroc,
                            "pro_auc" : pro_auc,
                            "epoch_i": i_epoch,
                            "epoch_p": i_epoch,
                            "epoch_pro": i_epoch,
                        }
                    )
                    result_I_collect.append(
                        {
                            "dataset_name": dataset_name,
                            "instance_auroc": auroc,
                            "full_pixel_auroc": full_pixel_auroc,
                            "pro_auc" : pro_auc,
                            "epoch": i_epoch,
                        }
                    )
                else:                   
                    if auroc > result_collect[-1]["instance_auroc"]:
                        result_collect[-1]["instance_auroc"] = auroc
                        result_collect[-1]["epoch_i"] = i_epoch
                        
                    if full_pixel_auroc > result_collect[-1]["full_pixel_auroc"]:
                        result_collect[-1]["full_pixel_auroc"] = full_pixel_auroc
                        result_collect[-1]["epoch_p"] = i_epoch
                        
                    if pro_auc >  result_collect[-1]["pro_auc"]:
                        result_collect[-1]["pro_auc"] = pro_auc
                        result_collect[-1]["epoch_pro"] = i_epoch
                        
                    if auroc > result_I_collect[-1]["instance_auroc"]:
                        result_I_collect[-1]["instance_auroc"] = auroc
                        result_I_collect[-1]["full_pixel_auroc"] = full_pixel_auroc
                        result_I_collect[-1]["pro_auc"] = pro_auc
                        result_I_collect[-1]["epoch_i"] = i_epoch      
                        
                    elif auroc == result_I_collect[-1]["instance_auroc"]:
                        if full_pixel_auroc > result_I_collect[-1]["full_pixel_auroc"]:
                            result_I_collect[-1]["full_pixel_auroc"] = full_pixel_auroc
                            result_I_collect[-1]["pro_auc"] = pro_auc
                            result_I_collect[-1]["epoch_i"] = i_epoch             
                       
                for key, item in result_collect[-1].items():
                    if key != "dataset_name":
                        LOGGER.info("{0}: {1:3.4f}".format(key, item))
                print(i_epoch)

        LOGGER.info("\n\n-----\n")
        with open(f'{patchcore_save_path}/result.txt', 'a') as f:
            f.write(f'result,{result_collect[-1]["instance_auroc"]},{result_collect[-1]["full_pixel_auroc"]},{result_collect[-1]["pro_auc"]},'\
                f'epoch:{result_collect[-1]["epoch_i"]+1}/{epochs},{result_collect[-1]["epoch_p"]+1}/{epochs},{result_collect[-1]["epoch_pro"]+1}/{epochs}') 
            f.write('\n')  
            f.write(f'result_I,{result_I_collect[-1]["instance_auroc"]},{result_I_collect[-1]["full_pixel_auroc"]},{result_I_collect[-1]["pro_auc"]},epoch:{result_I_collect[-1]["epoch"]+1}/{epochs}') 
    try:
        result_collect = [] 
        result_I_collect = []
        for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
            result_txt_path = os.path.join(run_save_path, "models",dataloaders["training"].name,"result.txt")
            for line in open(result_txt_path):
                lines = line.strip().split(",")
                if lines[0] == "result":
                    result_collect.append(
                        {
                            "dataset_name": dataloaders["training"].name,
                            "instance_auroc": float(lines[1]),
                            "full_pixel_auroc": float(lines[2]),
                            "pro_auc" : float(lines[3]),
                        }
                    ) 
                else:
                    result_I_collect.append(
                        {
                            "dataset_name": dataloaders["training"].name,
                            "instance_auroc": float(lines[1]),
                            "full_pixel_auroc": float(lines[2]),
                            "pro_auc" : float(lines[3]),
                        }
                    ) 
                    
        # Store a results and mean scores to a csv-file.
        result_metric_names = list(result_collect[-1].keys())[1:]
        result_dataset_names = [results["dataset_name"] for results in result_collect]
        result_scores = [list(results.values())[1:] for results in result_collect]
        patchcore.utils.compute_and_store_final_results(
            run_save_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_dataset_names,
        )
        
        result_metric_names = list(result_I_collect[-1].keys())[1:]
        result_dataset_names = [results["dataset_name"] for results in result_I_collect]
        result_scores = [list(results.values())[1:] for results in result_I_collect]
        patchcore.utils.compute_and_store_final_results(
            run_save_path,
            result_scores,
            column_names=result_metric_names,
            row_names=result_dataset_names,
            name="results_I"
        )
    except:
        pass
                
@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=["wideresnet50"])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=["layer2","layer3"])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=1)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, projection_lst, device):
        loaded_patchcores = []
        i=0
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                featuresprojection=projection_lst[i],
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
            i = i+1
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str,default="approx_greedy_coreset")
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)

@main.command("projection")
@click.option("--in_planes", type=int, default=1024)
@click.option("--out_planes", type=int, default=1024)
def projection(in_planes, out_planes):
    def get_projection(device):
        return patchcore.projection.Projection(in_planes, out_planes).to(device)
    return ("get_projection", get_projection)

@main.command("jigsawcls")
@click.option("--cls_num", type=int, default=4)
@click.option("--input_c", type=int, default=1024)
@click.option("--output_c", type=int, default=2048)
def jigsawcls(cls_num, input_c, output_c):
    def get_jigsawcls(device):
        return patchcore.projection.Jigsaw_cls(cls_num, input_c, output_c).to(device)
    return ("get_jigsawcls", get_jigsawcls)

@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--jig_batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=1, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    jig_batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            resize_ = 256 if subdataset == "transistor" else resize
            imagesize_ = 224 if subdataset == "transistor" else imagesize

            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize_,
                train_val_split=train_val_split,
                imagesize=imagesize_,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize_,
                imagesize=imagesize_,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            
            jigsaw_train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=jig_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "jigsaw_training": jigsaw_train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
