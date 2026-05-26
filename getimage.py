import os
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T

def create_horizontal_collage_from_two_dirs(dir1, dir2, selected_filenames1, selected_filenames2,
                                        output_path='128realimage_cwdata.jpg', target_size=(128, 128)):
    """
    改用 PIL 和 Torchvision 處理圖片，避免使用 cv2
    """
    
    # 準備轉換工具：縮放、中心裁剪(或填充)、轉為張量
    # 這裡使用 Compose 來確保圖片大小一致
    transform = T.Compose([
        T.Resize(target_size),
        T.CenterCrop(target_size),
        T.ToTensor()
    ])

    # 檢查檔案是否存在
    for d, files in [(dir1, selected_filenames1), (dir2, selected_filenames2)]:
        for filename in files:
            if not os.path.exists(os.path.join(d, filename)):
                raise FileNotFoundError(f"文件 {filename} 在目錄 {d} 中不存在")

    tensor_images = []

    # 處理所有選定的檔案
    all_files = [(dir1, f) for f in selected_filenames1] + [(dir2, f) for f in selected_filenames2]

    for directory, filename in all_files:
        img_path = os.path.join(directory, filename)
        # 使用 PIL 讀取
        img = Image.open(img_path).convert('RGB')
        # 轉換並加入列表
        tensor_images.append(transform(img))

    # 將所有圖像張量堆疊為一個批次
    batch = torch.stack(tensor_images)
    
    # 保存圖片
    save_image(batch, output_path, nrow=len(tensor_images))
    
    print(f"拼接圖已保存到 {output_path}")
    print(f"圖像尺寸: {target_size[1]}x{target_size[0]}, 數量: {len(tensor_images)}")

# 使用示例 (與原程式碼相同)
if __name__ == "__main__":
    dir1 = 'data/RS615_n'
    dir2 = 'data/RS615_n'
    selected_images1 = ['0_0090.jpg', '0_0135.jpg', '0_0180.jpg', '0_0225.jpg']
    selected_images2 = ['0_0270.jpg', '0_0315.jpg', '0_0000.jpg', '0_0045.jpg']

    # selected_images1 = ['0_0000.jpg', '0_0045.jpg', '0_0090.jpg', '0_0135.jpg']
    # selected_images2 = ['0_0180.jpg', '0_0225.jpg', '0_0270.jpg', '0_0315.jpg']

    target_size = (256, 256)
    
    create_horizontal_collage_from_two_dirs(
        dir1, dir2, 
        selected_images1, selected_images2,
        target_size=target_size,
        output_path='RS615.jpg'
    )