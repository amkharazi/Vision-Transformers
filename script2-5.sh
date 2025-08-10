# # ID009 - MNIST (10 classes)
# python train.py --run_id ID009_vit_tiny_tensorized_dummy_test_mnist --model_type tensorized --dataset mnist --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
# python test.py --run_id ID009_vit_tiny_tensorized_dummy_test_mnist --model_type tensorized --dataset mnist --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0

# # ID010 - CIFAR-10 (10 classes)
# python train.py --run_id ID010_vit_tiny_tensorized_dummy_test_cifar10 --model_type tensorized --dataset cifar10 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
# python test.py --run_id ID010_vit_tiny_tensorized_dummy_test_cifar10 --model_type tensorized --dataset cifar10 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0

# # ID011 - CIFAR-100 (100 classes)
# python train.py --run_id ID011_vit_tiny_tensorized_dummy_test_cifar100 --model_type tensorized --dataset cifar100 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 100 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
# python test.py --run_id ID011_vit_tiny_tensorized_dummy_test_cifar100 --model_type tensorized --dataset cifar100 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 100 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0

# # ID012 - TinyImageNet (200 classes)
# python train.py --run_id ID012_vit_tiny_tensorized_dummy_test_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 200 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
# python test.py --run_id ID012_vit_tiny_tensorized_dummy_test_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 256 --image_size 32 --patch_size 4 --num_classes 200 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0

# # ID013 - Fashion-MNIST (10 classes)
python train.py --run_id ID013_vit_tiny_tensorized_dummy_test_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
python test.py --run_id ID013_vit_tiny_tensorized_dummy_test_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0

# # ID014 - Flowers102 (102 classes)
# python train.py --run_id ID014_vit_tiny_tensorized_dummy_test_flowers102 --model_type tensorized --dataset flowers102 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 102 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
# python test.py --run_id ID014_vit_tiny_tensorized_dummy_test_flowers102 --model_type tensorized --dataset flowers102 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 102 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0

# # ID015 - Oxford Pets (37 classes)
# python train.py --run_id ID015_vit_tiny_tensorized_dummy_test_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 37 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
# python test.py --run_id ID015_vit_tiny_tensorized_dummy_test_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 256 --image_size 32 --patch_size 4 --num_classes 37 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0

# # ID016 - STL10 (10 classes)
# python train.py --run_id ID016_vit_tiny_tensorized_dummy_test_stl10 --model_type tensorized --dataset stl10 --batch_size 256 --epochs 600 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
# python test.py --run_id ID016_vit_tiny_tensorized_dummy_test_stl10 --model_type tensorized --dataset stl10 --batch_size 256 --image_size 32 --patch_size 4 --num_classes 10 --num_layers 6 --num_heads 2 2 2 --embed_dim 4 4 4 --mlp_dim 4 4 8 --tensor_method_mlp tle tle --tensor_method tle --reduce_level 0 0 0
