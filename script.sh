python train.py --test_id ID001_vit_original_tinyimagenet --dataset tinyimagenet --model_type original --num_layers 12 --epochs 100 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 16 16 4 --save_rate 20
python test.py  --test_id ID001_vit_original_tinyimagenet --dataset tinyimagenet --model_type original --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 16 16 4 --epochs 100 --save_rate 20

python train.py --test_id ID002_vit_tensorized_tle_tle_tinyimagenet --dataset tinyimagenet --model_type tensorized --num_layers 12 --epochs 100 --batch_size 64 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 16 16 4 --tensor_type tle tp --tdle_level 3 --save_rate 20
python test.py  --test_id ID002_vit_tensorized_tle_tle_tinyimagenet --dataset tinyimagenet --model_type tensorized --num_layers 12 --batch_size 16 --image_size 224 --patch_size 16 --embed_dim 16 16 3 --num_heads 2 2 3 --mlp_dim 16 16 4 --tensor_type tle tp --tdle_level 3 --epochs 100 --save_rate 20

