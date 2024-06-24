import pickle
import shutil
import os
from collections import Counter

pkl_file_path = '/Users/anguschen/Downloads/Create_JSON/texture_pkl/texture_img_test.pkl'
original_folder_path = '/Users/anguschen/Coding/VTON-CLIP/FashionTex_Changed(BEST)/data/texture_64'
target_folder_path = '/Users/anguschen/Downloads/Create_JSON/texture_img_test'
# Read and check .pkl file
with open(pkl_file_path, 'rb') as file:
    image_list = pickle.load(file)

# 使用Counter計算image_list中每個元素的出現次數
image_counter = Counter(image_list)
# 找出重複資料
duplicates = {item: count for item, count in image_counter.items() if count > 1}
# 如果有重複資料，印出重複資料及其數量
if duplicates:
    print("重複資料及其數量：")
    for item, count in duplicates.items():
        print(f"資料: {item}, 數量: {count}")
else:
    print("沒有重複資料。")
"""
重複資料及其數量：
資料: WOMEN-Dresses-id_00000747-01_7_additional.jpg, 數量: 2
資料: WOMEN-Dresses-id_00005701-03_2_side.jpg, 數量: 2
資料: WOMEN-Tees_Tanks-id_00002814-01_7_additional.jpg, 數量: 3
資料: WOMEN-Tees_Tanks-id_00002813-05_4_full.jpg, 數量: 2
資料: MEN-Tees_Tanks-id_00004154-01_7_additional.jpg, 數量: 2
資料: WOMEN-Blouses_Shirts-id_00003811-02_1_front.jpg, 數量: 2
資料: WOMEN-Skirts-id_00004513-06_4_full.jpg, 數量: 2
資料: WOMEN-Skirts-id_00002588-02_1_front.jpg, 數量: 2
資料: WOMEN-Blouses_Shirts-id_00000065-02_2_side.jpg, 數量: 2
資料: WOMEN-Cardigans-id_00003150-06_1_front.jpg, 數量: 2
資料: MEN-Tees_Tanks-id_00005834-09_7_additional.jpg, 數量: 2
資料: MEN-Jackets_Vests-id_00005451-01_4_full.jpg, 數量: 2
"""
# Copy Image from original_folder_path
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)
count = 0
for image_filename in image_list:
    source_image_path = os.path.join(original_folder_path, image_filename)
    target_image_path = os.path.join(target_folder_path, image_filename)
    shutil.copyfile(source_image_path, target_image_path)
    count+=1
print(len(image_list))
print(count, "圖片複製完成！")
"""
100
100 圖片複製完成！
"""

# Check number of images in target_folder_path
image_count = 0
for filename in os.listdir(target_folder_path):
    file_path = os.path.join(target_folder_path, filename)
    
    if os.path.isfile(file_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_count += 1
print("資料夾中的圖片數量為:", image_count)
"""
資料夾中的圖片數量為: 87
"""

