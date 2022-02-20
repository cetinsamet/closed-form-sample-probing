# validation run
python train.py --validation --gammaD 10 --gammaG 10 --gzsl --nclass_all 102 --latent_size 1024 --manualSeed 806 --syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --classifier_lr 0.001 --cuda --image_embedding res101 --recons_weight 0.01 --feedback_loop 1 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --dec_lr 0.0001 \
					--sample_probing --n_task 1 --n_probe_train 20 --n_probe_val 3 --k_probe_val 20 --n_subset 5 --alpha -3 --gamma 3 --n_syn 25 --sample_probing_loss_weight 1.0 --sample_probing_loss_type gzsl --closed_form_model_type sem2vis --lamb 0.001
# test run
python train.py --gammaD 10 --gammaG 10 --gzsl --nclass_all 102 --latent_size 1024 --manualSeed 806 --syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --classifier_lr 0.001 --cuda --image_embedding res101 --recons_weight 0.01 --feedback_loop 1 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --dec_lr 0.0001 \
					--sample_probing --n_task 1 --n_probe_train 20 --n_probe_val 3 --k_probe_val 20 --n_subset 5 --alpha -3 --gamma 3 --n_syn 25 --sample_probing_loss_weight 1.0 --sample_probing_loss_type gzsl --closed_form_model_type sem2vis --lamb 0.001