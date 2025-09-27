# # ID001 - MNIST (10 classes)
# python train.py --run_id ID001_vit_base_tensorized_mnist --model_type tensorized --dataset mnist --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id ID001_vit_base_tensorized_mnist --model_type tensorized --dataset mnist --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # ID002 - CIFAR-10 (10 classes)
# python train.py --run_id ID002_vit_base_tensorized_cifar10 --model_type tensorized --dataset cifar10 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id ID002_vit_base_tensorized_cifar10 --model_type tensorized --dataset cifar10 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # ID003 - CIFAR-100 (100 classes)
# python train.py --run_id ID003_vit_base_tensorized_cifar100 --model_type tensorized --dataset cifar100 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 100 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id ID003_vit_base_tensorized_cifar100 --model_type tensorized --dataset cifar100 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 100 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # ID004 - TinyImageNet (200 classes)
# python train.py --run_id ID004_vit_base_tensorized_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 200 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20 --train_size 70000 --repeat_count 1
python test.py  --run_id ID004_vit_base_tensorized_tinyimagenet --model_type tensorized --dataset tinyimagenet --batch_size 64 --image_size 224 --patch_size 16 --num_classes 200 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20 --train_size 70000 --repeat_count 1

# # ID005 - Fashion-MNIST (10 classes)
# python train.py --run_id ID005_vit_base_tensorized_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id ID005_vit_base_tensorized_fashionmnist --model_type tensorized --dataset fashionmnist --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # ID006 - Flowers102 (102 classes)
# python train.py --run_id ID006_vit_base_tensorized_flowers102 --model_type tensorized --dataset flowers102 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 102 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id ID006_vit_base_tensorized_flowers102 --model_type tensorized --dataset flowers102 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 102 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # ID007 - Oxford Pets (37 classes)
# python train.py --run_id ID007_vit_base_tensorized_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 37 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id ID007_vit_base_tensorized_oxford_pets --model_type tensorized --dataset oxford_pets --batch_size 64 --image_size 224 --patch_size 16 --num_classes 37 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle

# # ID008 - STL10 (10 classes)
# python train.py --run_id ID008_vit_base_tensorized_stl10 --model_type tensorized --dataset stl10 --batch_size 64 --epochs 600 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method tle --save_rate 20
# python test.py  --run_id ID008_vit_base_tensorized_stl10 --model_type tensorized --dataset stl10 --batch_size 64 --image_size 224 --patch_size 16 --num_classes 10 --num_layers 12 --num_heads 3 2 2 --embed_dim 3 16 16 --mlp_dim 3 32 32 --tensor_method_mlp tle tp --tensor_method