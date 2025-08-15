# # ID025 - MNIST (10 classes)
# python train.py --run_id ID025_vit_small_tensorized_test_mnist --model_type tensorized --dataset mnist --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID025_vit_small_tensorized_test_mnist --model_type tensorized --dataset mnist --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

# # ID026 - CIFAR-10 (10 classes)
# python train.py --run_id ID026_vit_small_tensorized_test_cifar10 --model_type tensorized --dataset cifar10 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID026_vit_small_tensorized_test_cifar10 --model_type tensorized --dataset cifar10 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

# # ID027 - CIFAR-100 (100 classes)
# python train.py --run_id ID027_vit_small_tensorized_test_cifar100 --model_type tensorized --dataset cifar100 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 100 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID027_vit_small_tensorized_test_cifar100 --model_type tensorized --dataset cifar100 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 100 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

# # ID028 - TinyImageNet (200 classes)
# python train.py --run_id ID028_vit_small_tensorized_test_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 200 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID028_vit_small_tensorized_test_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 256 --image_size 32 --patch_size 4 --num_classes 200 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

# # ID029 - Fashion-MNIST (10 classes)
# python train.py --run_id ID029_vit_small_tensorized_test_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID029_vit_small_tensorized_test_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

# # ID030 - Flowers102 (102 classes)
# python train.py --run_id ID030_vit_small_tensorized_test_flowers102 --model_type tensorized --dataset flowers102 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 102 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID030_vit_small_tensorized_test_flowers102 --model_type tensorized --dataset flowers102 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 102 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

# # ID031 - Oxford Pets (37 classes)
# python train.py --run_id ID031_vit_small_tensorized_test_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 37 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID031_vit_small_tensorized_test_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 256 --image_size 32 --patch_size 4 --num_classes 37 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

# # ID032 - STL10 (10 classes)
# python train.py --run_id ID032_vit_small_tensorized_test_stl10 --model_type tensorized --dataset stl10 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle
# python test.py  --run_id ID032_vit_small_tensorized_test_stl10 --model_type tensorized --dataset stl10 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 9 --num_heads 3 2 2 --embed_dim 3 8 8 --mlp_dim 6 8 8 --tensor_method_mlp tp tp --tensor_method tle

