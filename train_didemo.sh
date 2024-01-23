CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 32 \
--batch_size_val 32 \
--anno_path /mnt/nfs/CMG/zhanghaonan/datasets/Didemo/anns \
--video_path /mnt/nfs/CMG/zhanghaonan/datasets/Didemo/videos \
--datatype didemo \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir ckpt/didemo/main_exp/8query_intra_consistency_MSE_0.0001_inter_diversity_0.1margin_both_3cross_add_query_sim_query_shared_cross_att_shared_without_weight \
--center 1 \
--temp 3 \
--alpha 0.0001 \
--beta 0.005 \
--query_number 8 \
--base_encoder ViT-B/32 \
--cross_att_layer 3 \
--add_query_score_for_eval 0 \
--query_share 1 \
--cross_att_share 1 \