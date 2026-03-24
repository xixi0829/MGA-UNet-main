import glob
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from scipy import ndimage


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list=None, transform=None, mask_transform=None, train=True,
                 img_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.img_size = img_size

        if file_list is not None:
            self.valid_files = file_list
        else:
            self.valid_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                self.valid_files.extend(glob.glob(os.path.join(image_dir, ext)))

        # 减少打印干扰，只在初始化时打印一次即可
        # print(f"[{'Train' if train else 'Val'}] Loaded {len(self.valid_files)} images.")

    def get_mask_path(self, image_path):
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        mask_candidates = [
            os.path.join(self.mask_dir, name + '.png'),
            os.path.join(self.mask_dir, name + '.jpg'),
            os.path.join(self.mask_dir, name + '_segmentation.png'),
            os.path.join(self.mask_dir, name + '_mask.jpg'),
            os.path.join(self.mask_dir, name + '_lesion.png')
        ]
        for mask_path in mask_candidates:
            if os.path.exists(mask_path):
                return mask_path
        return os.path.join(self.mask_dir, name + '.png')

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        image_path = self.valid_files[idx]
        mask_path = self.get_mask_path(image_path)
        try:
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except Exception as e:
            print(f"Error loading file: {image_path}")
            image = Image.new('RGB', self.img_size)
            mask = Image.new('L', self.img_size)

        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)
        image = np.array(image)
        mask = np.array(mask)

        if self.train:
            image, mask = self.apply_augmentations(image, mask)

        image = self.normalize_image(image)
        mask = self.normalize_mask(mask)

        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

    def normalize_image(self, image):
        return image / 255.0

    def normalize_mask(self, mask):
        return np.where(mask > 128, 1.0, 0.0)

    def apply_augmentations(self, image, mask):
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        if random.random() > 0.5:
            pil_img = Image.fromarray(image.astype('uint8'), 'RGB')
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
            image = np.array(pil_img)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = ndimage.rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)
            mask = ndimage.rotate(mask, angle, reshape=False, order=0, mode='constant', cval=0)

        # 弹性变形
        if random.random() > 0.5:
            shape = image.shape
            alpha = random.uniform(30.0, 50.0)
            sigma = random.uniform(4.0, 6.0)
            dx = ndimage.gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = ndimage.gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            image_deformed = np.zeros_like(image)
            for i in range(3):
                image_deformed[:, :, i] = ndimage.map_coordinates(image[:, :, i], indices, order=1, mode='constant',
                                                                  cval=0).reshape(shape[:2])
            image = image_deformed
            mask = ndimage.map_coordinates(mask, indices, order=0, mode='constant', cval=0).reshape(shape[:2])

        return image, mask


class isic_loader(Dataset):
    """
    支持 K-Fold 交叉验证的数据加载器。
    添加 test_mode 参数用于测试时加载整个数据集
    """

    def __init__(self, path_Data, train=True, fold_idx=0, num_folds=5, seed=42, test_mode=False):
        super(isic_loader, self).__init__()

        full_image_dir = os.path.join(path_Data, 'images')
        full_mask_dir = os.path.join(path_Data, 'masks')

        # 1. 读取并排序
        all_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            all_files.extend(glob.glob(os.path.join(full_image_dir, ext)))
        all_files.sort()

        if len(all_files) == 0:
            raise Exception(f"No images found in {full_image_dir}")

        # 如果是测试模式，使用所有文件
        if test_mode:
            self.file_list = all_files
            print(f"[Test Mode] Total test images: {len(self.file_list)}")
        else:
            # 2. 随机打乱
            random.seed(seed)
            random.shuffle(all_files)

            # 3. K-Fold 切分逻辑
            total_files = len(all_files)
            fold_size = total_files // num_folds

            # 每一份的起始和结束索引
            val_start = fold_idx * fold_size
            if fold_idx == num_folds - 1:
                val_end = total_files  # 最后一折包含所有剩余图片
            else:
                val_end = (fold_idx + 1) * fold_size

            val_files = all_files[val_start:val_end]
            train_files = all_files[:val_start] + all_files[val_end:]

            # 4. 根据模式选择文件
            if train:
                self.file_list = train_files
                print(f"[Fold {fold_idx}] Train Data: {len(self.file_list)}")
            else:
                self.file_list = val_files
                print(f"[Fold {fold_idx}] Val Data:   {len(self.file_list)}")

        # 5. 实例化
        self.dataset = ImageMaskDataset(
            image_dir=full_image_dir,
            mask_dir=full_mask_dir,
            file_list=self.file_list,
            train=train if not test_mode else False,  # 测试时不使用数据增强
            img_size=(256, 256)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]