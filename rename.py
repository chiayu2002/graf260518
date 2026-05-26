import os
import shutil  # 用於複製檔案

# 指定資料夾路徑
source_folder = "/Data/home/vicky/graf250311/data/rs615"  # 原始資料夾
target_folder = "/Data/home/vicky/graf250311/data/RS615_long"  # 新資料夾

# 建立目標資料夾（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# 先列出資料夾中的所有檔案
print("資料夾中的檔案：")
files = os.listdir(source_folder)
for file in files:
    print(file)

print("\n開始處理檔案...")

# 創建重命名映射
rename_map = {}
for file in files:
    try:
        if '_' in file:
            # 分離檔名和副檔名
            name_part, ext = os.path.splitext(file)
            prefix, num_str = name_part.split('_')
            num = int(num_str)
            
            # 所有檔案都加180度，如果大於等於360度就減360
            new_num = (num + 180) % 360
            new_name = f"{prefix}_{str(new_num).zfill(4)}{ext}"
            
            old_path = os.path.join(source_folder, file)
            new_path = os.path.join(target_folder, new_name)
            
            print(f"計劃重命名: {file} -> {new_name}")
            rename_map[old_path] = new_path
    except Exception as e:
        print(f"處理檔案 {file} 時發生錯誤: {str(e)}")

if not rename_map:
    print("沒有找到需要重命名的檔案！")
    exit()

# 直接複製到新資料夾並重命名
for old_path, new_path in rename_map.items():
    try:
        shutil.copy2(old_path, new_path)  # copy2 保留檔案的metadata
        print(f"成功: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
    except Exception as e:
        print(f"複製失敗: {os.path.basename(old_path)}, 錯誤: {str(e)}")

print("\n複製和重命名完成！")