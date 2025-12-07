python train_original_version2_adamw.py --run_id ID000_vit_original_version_2 --dataset tinyimagenet --batch_size 32 --epochs 200 --image_size 224 --num_classes 200
python test_original_version2_adamw.py --run_id ID000_vit_original_version_2 --dataset tinyimagenet --batch_size 32 --image_size 224 --num_classes 200

python train_tensor_4_version1_adamw.py --run_id ID004_vit_tensor_v4_version_2 --dataset tinyimagenet --batch_size 32 --epochs 200 --image_size 224 --num_classes 200
python test_tensor_4_version1_adamw.py --run_id ID004_vit_tensor_v4_version_2 --dataset tinyimagenet --batch_size 32 --image_size 224 --num_classes 200

python train_tensor_6_version1_adamw.py --run_id ID005_vit_tensor_v6_version_2 --dataset tinyimagenet --batch_size 32 --epochs 200 --image_size 224 --num_classes 200
python test_tensor_6_version1_adamw.py --run_id ID005_vit_tensor_v6_version_2 --dataset tinyimagenet --batch_size 32 --image_size 224 --num_classes 200
