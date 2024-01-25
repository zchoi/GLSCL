CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 \
python -m torch.distributed.launch \
--master_port 2503 \
--nproc_per_node=4 \
main_retrieval.py \
--do_eval 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path /mnt/nfs/CMG/zhanghaonan/datasets/activitynet/anns \
--video_path /mnt/nfs/CMG/zhanghaonan/datasets/activitynet/Videos/Activity_Videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--output_dir ckpt/activitynet/main_exp/8query_intra_consistency_MSE_0.0001_inter_diversity_0.1margin_both_3cross_add_query_sim_query_shared_cross_att_shared_without_weight \
--center 1 \
--query_number 8 \
--cross_att_layer 3 \
--query_share 1 \
--cross_att_share 1 \
--add_query_score_for_eval 0 \
--base_encoder ViT-B/32 \
--temp 3 \
--alpha 0.0001 \
--beta 0.005 \
--t2v_beta 50 \
--v2t_beta 50 \
--init_model ckpt/activitynet/main_exp/8query_intra_consistency_MSE_0.0001_inter_diversity_0.1margin_both_3cross_add_query_sim_query_shared_cross_att_shared_without_weight/pytorch_model.bin.step900.5