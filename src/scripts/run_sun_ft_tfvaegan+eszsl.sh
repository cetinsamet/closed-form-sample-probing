# validation run
python train.py --validation --finetune --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 1 --feed_lr 0.0001 \
					--sample_probing --n_task 1 --n_probe_train 30 --n_probe_val 15 --k_probe_val 10 --n_subset 5 --alpha -3 --gamma 3 --n_syn 35 --sample_probing_loss_weight 5.0 --sample_probing_loss_type zsl --closed_form_model_type eszsl
# test run
python train_tfvaegan.py --finetune --gammaD 1 --gammaG 1 --gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 1 --feed_lr 0.0001 \
					--sample_probing --n_task 1 --n_probe_train 30 --n_probe_val 15 --k_probe_val 10 --n_subset 5 --alpha -3 --gamma 3 --n_syn 35 --sample_probing_loss_weight 5.0 --sample_probing_loss_type zsl --closed_form_model_type eszsl