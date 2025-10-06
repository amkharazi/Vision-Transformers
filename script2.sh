python train.py --test_id ID001_vit_original_tinyimagenet_V2 --dataset tinyimagenet --model_type original --num_layers 12 --epochs 100 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --save_rate 20 --repeat_count 1
python test.py  --test_id ID001_vit_original_tinyimagenet_V2 --dataset tinyimagenet --model_type original --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --epochs 100 --save_rate 20 --repeat_count 1

python train.py --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_V2 --dataset tinyimagenet --model_type tensorized --num_layers 12 --epochs 100 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --save_rate 20 --repeat_count 1
python test.py  --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_V2 --dataset tinyimagenet --model_type tensorized --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --epochs 100 --save_rate 20 --repeat_count 1

python train.py --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V2 --dataset tinyimagenet --model_type tensorized --num_layers 12 --epochs 100 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --weight_decay 0 --save_rate 20 --repeat_count 1
python test.py  --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V2 --dataset tinyimagenet --model_type tensorized --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --epochs 100 --save_rate 20 --repeat_count 1

# Change the betas and lr back 0.9 0.98 5e-4
python train.py --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V3 --dataset tinyimagenet --model_type tensorized --num_layers 12 --epochs 100 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --weight_decay 0 --save_rate 20 --repeat_count 1
python test.py  --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V3 --dataset tinyimagenet --model_type tensorized --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --epochs 100 --save_rate 20 --repeat_count 1

python train.py --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V4 --dataset tinyimagenet --model_type tensorized --num_layers 12 --epochs 600 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --weight_decay 0 --save_rate 20 --repeat_count 1
python test.py  --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V4 --dataset tinyimagenet --model_type tensorized --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --epochs 600 --save_rate 20 --repeat_count 1

python train.py --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V5 --dataset tinyimagenet --model_type tensorized --num_layers 12 --epochs 100 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --weight_decay 0 --save_rate 20 --repeat_count 1 --num_tensorized 2
python test.py  --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_nowd_V5 --dataset tinyimagenet --model_type tensorized --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 32 32 3 --tensor_type tle tp --tdle_level 3 --epochs 100 --save_rate 20 --repeat_count 1 --num_tensorized 2

## SMALL version on cifar100

python train.py --test_id ID001_vit_original_tinyimagenet_adam_small --dataset cifar100 --model_type original --num_layers 9 --epochs 100 --batch_size 64 --image_size 32 --patch_size 4 --embed_dim 4 4 3 --num_heads 2 2 3 --mlp_dim 8 8 3 --save_rate 20 --repeat_count 1 --type adam
python test.py  --test_id ID001_vit_original_tinyimagenet_adam_small --dataset cifar100 --model_type original --num_layers 9 --batch_size 64 --image_size 32 --patch_size 4 --embed_dim 4 4 3 --num_heads 2 2 3 --mlp_dim 8 8 3 --epochs 100 --save_rate 20 --repeat_count 1 --type adam

python train.py --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_adam_small --dataset cifar100 --model_type tensorized --num_layers 9 --epochs 100 --batch_size 64 --image_size 32 --patch_size 4 --embed_dim 4 4 3 --num_heads 2 2 3 --mlp_dim 8 8 3 --tensor_type tle tp --tdle_level 3 --save_rate 20 --repeat_count 1 --type adam
python test.py  --test_id ID002_vit_tensorized_tle_tp_tinyimagenet_adam_small --dataset cifar100 --model_type tensorized --num_layers 9 --batch_size 64 --image_size 32 --patch_size 4 --embed_dim 4 4 3 --num_heads 2 2 3 --mlp_dim 8 8 3 --tensor_type tle tp --tdle_level 3 --epochs 100 --save_rate 20 --repeat_count 1 --type adam
