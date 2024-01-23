CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 \
python -m torch.distributed.launch \
--master_port 2513 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 64 \
--anno_path /mnt/nfs/CMG/zhanghaonan/datasets/MSR-VTT/anns \
--video_path /mnt/nfs/CMG/zhanghaonan/datasets/MSR-VTT/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir ckpt/msrvtt/no_qbnorm/alpha0.0001_beta_0.02 \
--center 1 \
--temp 3 \
--alpha 0.0001 \
--beta 0.02 \
--query_number 8 \
--base_encoder ViT-B/32 \
--cross_att_layer 3 \
--query_share 1 \
--cross_att_share 1 \
--loss2_weight 0.5 \