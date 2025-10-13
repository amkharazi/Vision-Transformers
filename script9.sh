# # EXPID001 - MNIST (10 classes)
# python train.py --run_id EXPID001_vit_small_tensorized_test_mnist --model_type tensorized --dataset mnist --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id EXPID001_vit_small_tensorized_test_mnist --model_type tensorized --dataset mnist --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # EXPID002 - CIFAR-10 (10 classes)
# python train.py --run_id EXPID002_vit_small_tensorized_test_cifar10 --model_type tensorized --dataset cifar10 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id EXPID002_vit_small_tensorized_test_cifar10 --model_type tensorized --dataset cifar10 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # EXPID003 - CIFAR-100 (100 classes)
# python train.py --run_id EXPID003_vit_small_tensorized_test_cifar100 --model_type tensorized --dataset cifar100 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 100 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id EXPID003_vit_small_tensorized_test_cifar100 --model_type tensorized --dataset cifar100 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 100 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # EXPID004 - TinyImageNet (200 classes)
# python train.py --run_id EXPID004_vit_small_tensorized_test_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 200 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
python test.py  --run_id EXPID004_vit_small_tensorized_test_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 64 --image_size 224 --patch_size 16 --num_classes 200 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # EXPID005 - Fashion-MNIST (10 classes)
# python train.py --run_id EXPID005_vit_small_tensorized_test_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id EXPID005_vit_small_tensorized_test_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # EXPID006 - Flowers102 (102 classes)
# python train.py --run_id EXPID006_vit_small_tensorized_test_flowers102 --model_type tensorized --dataset flowers102 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 102 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id EXPID006_vit_small_tensorized_test_flowers102 --model_type tensorized --dataset flowers102 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 102 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # EXPID007 - Oxford Pets (37 classes)
# python train.py --run_id EXPID007_vit_small_tensorized_test_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 37 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id EXPID007_vit_small_tensorized_test_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 64 --image_size 224 --patch_size 16 --num_classes 37 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # EXPID008 - STL10 (10 classes)
# python train.py --run_id EXPID008_vit_small_tensorized_test_stl10 --model_type tensorized --dataset stl10 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id EXPID008_vit_small_tensorized_test_stl10 --model_type tensorized --dataset stl10 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method