from PIL import Image
import os
import glob
import cv2
import numpy as np

def create_video(image_folder, output_path='output.mp4', fps=10, loop=0, prefix='', resize=None, start_number=0, max_frames=None):
    """
    將資料夾中的圖片轉換成 MP4 影片
    
    參數:
        image_folder: 圖片所在的資料夾路徑
        output_path: 輸出的影片檔案路徑（預設為 'output.mp4'）
        fps: 每秒幀數 (預設 10)
        loop: 循環次數，0 表示不循環（預設 0），大於 0 表示重複次數
        prefix: 檔名前綴過濾，例如 '0_' 只處理 0_ 開頭的圖片
        resize: 調整圖片大小，可以是單一數字(如256)或tuple(如(256,256))
        start_number: 開始的數字，例如 180 表示從 0_0180 開始
        max_frames: 最多使用的圖片數量，例如 180 表示只使用 180 張
    """
    
    # 支援的圖片格式
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    
    # 取得所有圖片檔案
    image_files = []
    for ext in extensions:
        pattern = f"{prefix}*{ext}" if prefix else ext
        image_files.extend(glob.glob(os.path.join(image_folder, pattern)))
        image_files.extend(glob.glob(os.path.join(image_folder, pattern.upper())))
    
    # 排序檔案名稱
    image_files.sort(key=lambda x: os.path.basename(x))
    
    # 篩選出偶數的圖片（例如：0_0000, 0_0002, 0_0004...）
    if prefix:
        files_after_start = []
        files_before_start = []
        
        for file in image_files:
            basename = os.path.basename(file)
            try:
                name_without_ext = os.path.splitext(basename)[0]
                number_part = name_without_ext.split('_')[-1]
                number = int(number_part)
                
                # 只保留偶數
                if number % 2 == 0:
                    if number >= start_number:
                        files_after_start.append(file)
                    else:
                        files_before_start.append(file)
            except (ValueError, IndexError):
                continue
        
        image_files = files_after_start + files_before_start
    
    # 限制圖片數量
    if max_frames and len(image_files) > max_frames:
        image_files = image_files[:max_frames]
        print(f"限制為前 {max_frames} 張圖片")
    
    if not image_files:
        print(f"在 {image_folder} 中找不到符合條件的圖片檔案")
        if prefix:
            print(f"(搜尋前綴: {prefix}，且檔名數字為偶數)")
        return
    
    print(f"找到 {len(image_files)} 張符合條件的圖片（前綴: '{prefix}'，偶數）")
    print(f"播放順序: 先從 {start_number} 開始，然後接 0 到 {start_number-2}")
    
    # 如果需要循環，複製圖片列表
    if loop > 0:
        image_files = image_files * (loop + 1)
        print(f"設定循環 {loop} 次，總共 {len(image_files)} 張圖片")
    
    # 讀取第一張圖片來確定尺寸
    first_img = Image.open(image_files[0])
    
    if resize:
        if isinstance(resize, int):
            target_size = (resize, resize)
        else:
            target_size = resize
        width, height = target_size
    else:
        width, height = first_img.size
    
    print(f"影片尺寸: {width}x{height}, FPS: {fps}")
    
    # 設定影片編碼器和輸出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'avc1', 'H264'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 處理每張圖片
    for i, filename in enumerate(image_files):
        try:
            img = Image.open(filename)
            
            # 調整圖片大小
            if resize:
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # 轉換為 RGB 模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 轉換為 numpy array 並改變顏色順序 (RGB -> BGR for OpenCV)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 寫入影片
            out.write(frame)
            
            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"已處理: {i + 1}/{len(image_files)} 張圖片")
                
        except Exception as e:
            print(f"無法處理 {filename}: {e}")
    
    # 釋放資源
    out.release()
    
    print(f"\n成功建立 MP4: {output_path}")
    print(f"總共 {len(image_files)} 張圖片，FPS: {fps}")
    print(f"影片長度: {len(image_files)/fps:.2f} 秒")


def create_gif(image_folder, output_path='output.gif', duration=500, loop=0, prefix='', resize=None, start_number=0):
    """
    將資料夾中的圖片轉換成 GIF
    
    參數:
        image_folder: 圖片所在的資料夾路徑
        output_path: 輸出的 GIF 檔案路徑（預設為 'output.gif'）
        duration: 每張圖片顯示的時間，單位為毫秒（預設 500ms）
        loop: 循環次數，0 表示無限循環（預設 0）
        prefix: 檔名前綴過濾，例如 '0_' 只處理 0_ 開頭的圖片
        resize: 調整圖片大小，可以是單一數字(如256)或tuple(如(256,256))
        start_number: 開始的數字，例如 180 表示從 0_0180 開始
    """
    
    # 支援的圖片格式
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
    
    # 取得所有圖片檔案
    image_files = []
    for ext in extensions:
        pattern = f"{prefix}*{ext}" if prefix else ext
        image_files.extend(glob.glob(os.path.join(image_folder, pattern)))
        image_files.extend(glob.glob(os.path.join(image_folder, pattern.upper())))
    
    # 排序檔案名稱
    image_files.sort(key=lambda x: os.path.basename(x))
    
    # 篩選出 5 的倍數的圖片
    if prefix:
        files_after_start = []
        files_before_start = []
        
        for file in image_files:
            basename = os.path.basename(file)
            try:
                name_without_ext = os.path.splitext(basename)[0]
                number_part = name_without_ext.split('_')[-1]
                number = int(number_part)
                
                if number % 5 == 0:
                    if number >= start_number:
                        files_after_start.append(file)
                    else:
                        files_before_start.append(file)
            except (ValueError, IndexError):
                continue
        
        image_files = files_after_start + files_before_start
    
    if not image_files:
        print(f"在 {image_folder} 中找不到符合條件的圖片檔案")
        if prefix:
            print(f"(搜尋前綴: {prefix}，且檔名數字為 5 的倍數)")
        return
    
    print(f"找到 {len(image_files)} 張符合條件的圖片（前綴: '{prefix}'，5的倍數）")
    print(f"播放順序: 先從 {start_number} 開始，然後接 0 到 {start_number-5}")
    
    # 載入所有圖片
    images = []
    for filename in image_files:
        try:
            img = Image.open(filename)
            
            # 調整圖片大小
            if resize:
                if isinstance(resize, int):
                    target_size = (resize, resize)
                else:
                    target_size = resize
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 轉換為 RGB 模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            images.append(img)
            print(f"已載入: {os.path.basename(filename)}")
        except Exception as e:
            print(f"無法載入 {filename}: {e}")
    
    if not images:
        print("沒有成功載入任何圖片")
        return
    
    # 儲存為 GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    
    print(f"\n成功建立 GIF: {output_path}")
    print(f"總共 {len(images)} 張圖片，每張顯示 {duration}ms")
    if resize:
        print(f"圖片大小: {images[0].size}")


# 使用範例
if __name__ == "__main__":
    image_folder = '/Data/home/vicky/graf260108_im64/data/RS307_n'
    
    # 選擇輸出格式：MP4 或 GIF
    output_format = 'mp4'  # 改成 'gif' 可以輸出 GIF
    
    if output_format == 'mp4':
        # 製作 MP4 影片 - 取偶數圖片，總共 180 張
        create_video(
            image_folder=image_folder,
            output_path='307_animation.mp4',
            fps=25,                    # 每秒 10 幀
            loop=0,                    # 0 表示不循環
            prefix='0_',
            resize=256,
            start_number=180,
            max_frames=180             # 只使用 180 張圖片
        )
    else:
        # 製作 GIF
        create_gif(
            image_folder=image_folder,
            output_path='0_animation.gif',
            duration=100,              # 每張圖 100 毫秒
            loop=0,                    # 0 表示無限循環
            prefix='0_',
            resize=256,
            start_number=0
        )