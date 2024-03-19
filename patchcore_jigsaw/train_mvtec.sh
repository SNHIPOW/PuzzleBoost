datapath=dataset/mvtec
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

### IM224:
# Performance: Instance AUROC: 0.9963, Pixelwise AUROC: 0.9814, PRO: 0.9435
python run_jigsaw_img.py --gpu 0 --epochs 40 --seed 0 --save_patchcore_model \
--log_group sd0_ep40_bs2_n4_p0.01_ps3_nn1 --log_project re50_size-224_num-4 result_mvtec \
patch_core -b wideresnet50 -le layer2 -le layer3  --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.01 approx_greedy_coreset \
dataset --jig_batch_size 2 --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath \
projection --in_planes 1024 --out_planes 1024 \
jigsawcls --cls_num 4 --input_c 1024 --output_c 2048