python original.py --TEST_ID vit_tiny_dummy_test --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128
python original_val.py --TEST_ID vit_tiny_dummy_test --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128

python original.py --TEST_ID vit_small_dummy_test --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 12 --embed_dim 192 --mlp_dim 384
python original_val.py --TEST_ID vit_small_dummy_test --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 12 --embed_dim 192 --mlp_dim 384

python tensorized.py --TEST_ID vit_tiny_tensorized_dummy_test_tle --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128 --tensor_method_mlp tle tle --tensor_method tle
python tensorized_val.py --TEST_ID vit_tiny_tensorized_dummy_test_tle --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128 --tensor_method_mlp tle tle --tensor_method tle

python tensorized.py --TEST_ID vit_tiny_tensorized_dummy_test_tdle --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128 --tensor_method_mlp tdle tdle --tensor_method tdle
python tensorized_val.py --TEST_ID vit_tiny_tensorized_dummy_test_tdle --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128 --tensor_method_mlp tdle tdle --tensor_method tdle

python tensorized.py --TEST_ID vit_tiny_tensorized_dummy_test_tle_tp --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128 --tensor_method_mlp tle tp --tensor_method tle
python tensorized_val.py --TEST_ID vit_tiny_tensorized_dummy_test_tle_tp --dataset mnist --batch_size 256 --n_epoch 5 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 8 --embed_dim 64 --mlp_dim 128 --tensor_method_mlp tle tp --tensor_method tle
