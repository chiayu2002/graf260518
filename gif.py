from PIL import Image
import os
import glob
import numpy as np
import imageio


def create_video(image_folder, output_path='output.mp4', fps=25, loop=0,
                 prefix='', resize=None, start_number=0, max_frames=None):
    """
    將資料夾中的圖片轉換成 MP4 影片（使用 imageio，不需要 cv2）

    參數:
        image_folder: 圖片所在的資料夾路徑
        output_path:  輸出的影片檔案路徑（預設為 'output.mp4'）
        fps:          每秒幀數（預設 25）
        loop:         循環次數，0 表示不循環
        prefix:       檔名前綴過濾，例如 '0_' 只處理 0_ 開頭的圖片
        resize:       調整圖片大小，可以是單一數字(如256)或tuple(如(256,256))
        start_number: 從哪個編號開始（先播 start_number 之後，再接前面）
        max_frames:   最多使用的圖片數量
    """
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

    image_files = []
    for ext in extensions:
        pattern = f"{prefix}*{ext}" if prefix else ext
        image_files.extend(glob.glob(os.path.join(image_folder, pattern)))
        image_files.extend(glob.glob(os.path.join(image_folder, pattern.upper())))

    image_files.sort(key=lambda x: os.path.basename(x))

    # 每 5 張取一張（5 的倍數）
    if prefix:
        files_after_start  = []
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

    # 限制圖片數量
    if max_frames and len(image_files) > max_frames:
        image_files = image_files[:max_frames]
        print(f"限制為前 {max_frames} 張圖片")

    if not image_files:
        print(f"在 {image_folder} 中找不到符合條件的圖片檔案")
        return

    print(f"找到 {len(image_files)} 張符合條件的圖片（前綴: '{prefix}'，5的倍數）")
    print(f"播放順序: 先從 {start_number} 開始，然後接 0 到 {start_number - 5}")

    # 循環
    if loop > 0:
        image_files = image_files * (loop + 1)
        print(f"設定循環 {loop} 次，總共 {len(image_files)} 張圖片")

    # 決定輸出尺寸
    first_img = Image.open(image_files[0])
    if resize:
        target_size = (resize, resize) if isinstance(resize, int) else tuple(resize)
    else:
        target_size = first_img.size  # (width, height)

    width, height = target_size
    print(f"影片尺寸: {width}x{height}, FPS: {fps}")

    # 用 imageio 寫入 mp4
    writer = imageio.get_writer(output_path, fps=fps, quality=8, format='FFMPEG')

    for i, filename in enumerate(image_files):
        try:
            img = Image.open(filename).convert('RGB')
            if resize:
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            frame = np.array(img)   # RGB，imageio 直接接受
            writer.append_data(frame)

            if (i + 1) % 10 == 0 or i == len(image_files) - 1:
                print(f"已處理: {i + 1}/{len(image_files)} 張圖片")

        except Exception as e:
            print(f"無法處理 {filename}: {e}")

    writer.close()

    print(f"\n成功建立 MP4: {output_path}")
    print(f"總共 {len(image_files)} 張圖片，FPS: {fps}")
    print(f"影片長度: {len(image_files)/fps:.2f} 秒")


def create_gif(image_folder, output_path='output.gif', duration=100, loop=0,
               prefix='', resize=None, start_number=0, max_frames=None):
    """
    將資料夾中的圖片轉換成 GIF

    參數:
        duration: 每張圖片顯示的時間，單位為毫秒（預設 100ms）
        loop:     0 表示無限循環
    """
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']

    image_files = []
    for ext in extensions:
        pattern = f"{prefix}*{ext}" if prefix else ext
        image_files.extend(glob.glob(os.path.join(image_folder, pattern)))
        image_files.extend(glob.glob(os.path.join(image_folder, pattern.upper())))

    image_files.sort(key=lambda x: os.path.basename(x))

    if prefix:
        files_after_start  = []
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

    if max_frames and len(image_files) > max_frames:
        image_files = image_files[:max_frames]

    if not image_files:
        print(f"在 {image_folder} 中找不到符合條件的圖片檔案")
        return

    print(f"找到 {len(image_files)} 張圖片")

    images = []
    for filename in image_files:
        try:
            img = Image.open(filename).convert('RGB')
            if resize:
                target_size = (resize, resize) if isinstance(resize, int) else tuple(resize)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)
            print(f"已載入: {os.path.basename(filename)}")
        except Exception as e:
            print(f"無法載入 {filename}: {e}")

    if not images:
        print("沒有成功載入任何圖片")
        return

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


# ── 使用範例 ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_folder = '/Data/home/vicky/graf260108_im64/data/RS615_n'

    output_format = 'mp4'   # 改成 'gif' 輸出 GIF

    if output_format == 'mp4':
        create_video(
            image_folder=image_folder,
            output_path='615_animation.mp4',
            fps=25,
            loop=0,
            prefix='0_',
            resize=256,
            start_number=0,
            max_frames=72
        )
    else:
        create_gif(
            image_folder=image_folder,
            output_path='0_animation.gif',
            duration=100,
            loop=0,
            prefix='0_',
            resize=256,
            start_number=0
        )