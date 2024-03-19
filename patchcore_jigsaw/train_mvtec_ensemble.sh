datapath=dataset/mvtec
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

### IM480:
# Performance: Instance AUROC: 0.9977, Pixelwise AUROC: 0.9869, PRO: 0.9659
python run_jigsaw_ensemble.py --gpu 0 --epochs 40 --seed 40 --save_patchcore_model \
--log_group s40_ep40_bs2_n3_p0.005_ps3_nn1 --log_project embed_size-480_num-3 result_mvtec \
patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 \
--faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.005 approx_greedy_coreset \
dataset --jig_batch_size 2 --resize 512 --imagesize 480 "${dataset_flags[@]}" mvtec $datapath \
projection --in_planes 384 --out_planes 384 --n_layers 1 --layer_type 0 \
jigsawcls --cls_num 3 --input_c 384 --output_c 768