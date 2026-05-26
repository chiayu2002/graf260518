# from PIL import Image
# import numpy as np

# # 請換成您的一張訓練圖片路徑
# filename = "/Data/home/vicky/graf260108_im64/data/RS307_n/0_0000.jpg" 

# # 方法 A: 原本的寫法
# img_old = Image.open(filename).convert('RGB')
# pixel_old = img_old.getpixel((0, 0)) # 假設左上角是背景

# # 方法 B: 新的強制黑底寫法
# img_rgba = Image.open(filename).convert('RGBA')
# bg = Image.new('RGBA', img_rgba.size, (0, 0, 0))
# img_new = Image.alpha_composite(bg, img_rgba).convert('RGB')
# pixel_new = img_new.getpixel((0, 0))

# print(f"原本寫法背景顏色: {pixel_old}")
# print(f"新寫法背景顏色: {pixel_new}")

# # 檢查是否完全一樣
# if np.array_equal(np.array(img_old), np.array(img_new)):
#     print("結論：兩者像素完全一致，您運氣很好，不改也沒關係。")
# else:
#     print("結論：兩者有差異！原本的寫法可能含有雜訊，請務必修改程式碼！")


from PIL import Image
import numpy as np

# 設定您的圖片路徑
filename = "/Data/home/vicky/graf260108_im64/data/RS307_n/0_0180.jpg"

def clean_background_aggressive(image_path, threshold=60):
    """
    更積極的去背：只要像素夠暗（小於 threshold），就視為背景雜訊，強制轉為純黑。
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)

    # 1. 找出真正的背景顏色 (Sampling)
    # 我們取圖片四個角落，看看哪個最暗，當作參考
    h, w, _ = arr.shape
    corners = [arr[0, 0], arr[0, w-1], arr[h-1, 0], arr[h-1, w-1]]
    # 計算亮度 (平均 RGB)
    brightness = [np.mean(c) for c in corners]
    min_idx = np.argmin(brightness)
    darkest_corner_val = corners[min_idx]
    
    print(f"偵測到最暗的角落 (可能是背景): {darkest_corner_val} (亮度: {brightness[min_idx]:.1f})")

    if brightness[min_idx] > 100:
        print("警告：四個角落都很亮，這張圖可能填滿了柱子，或者背景不是黑的！")

    # 2. 執行去背 (Thresholding)
    # 邏輯：只要 RGB 三個通道都小於 threshold (例如 60)，就變成 (0,0,0)
    # 您的柱子是 (190, 190, 190)，非常安全，不會被刪掉。
    mask = (arr[:, :, 0] < threshold) & (arr[:, :, 1] < threshold) & (arr[:, :, 2] < threshold)
    
    # 計算有多少像素被修改了
    changed_pixels = np.sum(mask)
    total_pixels = w * h
    print(f"修正了 {changed_pixels} 個像素 ({changed_pixels/total_pixels*100:.1f}%) -> 變成純黑")

    arr[mask] = [0, 0, 0]
    return Image.fromarray(arr)

# --- 執行測試 ---
print("正在分析圖片...")
# 建議閾值設為 50~60 (因為您的背景看起來像深灰，可能在 30~50 之間)
img_clean = clean_background_aggressive(filename, threshold=65)

# 儲存結果讓您檢查
save_path = "test_clean_black.jpg"
img_clean.save(save_path)
print(f"已儲存測試圖片至: {save_path}，請打開檢查背景是否變全黑，且柱子陰影還在。")