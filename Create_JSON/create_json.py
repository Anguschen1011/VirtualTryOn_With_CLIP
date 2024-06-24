import os
import json
import random
from tqdm import tqdm

def gender_to_text(filename):
    if "WOMEN" in filename:
        num = random.randint(1,100)
        if num%2 == 0:
            dress_options = random.randint(1,100)
            if dress_options %2 == 0:
                text = [["tank dress","short dress"],["tank short dress"]]
            else:
                text = [["tank dress","long dress"],["tank long dress"]]
        else:
            shirt_options = ["short shirt", "long shirt"]
            pants_options = ["shorts", "long pants"]
            text = ["", [random.choice(shirt_options), random.choice(pants_options)]]
    elif "MEN" in filename:
        shirt_options = ["short shirt", "long shirt"]
        pants_options = ["shorts", "long pants"]
        text = ["", [random.choice(shirt_options), random.choice(pants_options)]]
    else:
        text = None
    return text

def get_random_texture(image_files, num_images=2):
    texture_images = []
    random_images = random.sample(image_files, num_images)
    for image_name in random_images:
        texture_images.append(image_name)
    return texture_images

def process_images(folder_path, texture_path):
    count = 0
    data = []
    image_files = [filename for filename in os.listdir(texture_path) if filename.endswith(('.jpg'))]
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".png"):
            #Image Name
            img_name = os.path.splitext(filename)[0]
            #Image Text
            text = gender_to_text(filename)
            #Image Texture
            if len(image_files) < 2:
                image_files = [filename for filename in os.listdir(texture_path) if filename.endswith('.jpg')]
            texture = get_random_texture(image_files)
            for image_name in texture:
                image_files.remove(image_name)
            img_data = {
                "img": img_name,
                "text":text,
                "texture": texture
            }
            data.append(img_data)
            count +=1
    print("Processing complete for",  count ,"images.")
    return data

def check_img_num(target_folder_path):
    image_count = 0
    for filename in os.listdir(target_folder_path):
        file_path = os.path.join(target_folder_path, filename)
        
        if os.path.isfile(file_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_count += 1
    print("Number of images in the folder :", image_count)

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    folder_path = "/Users/anguschen/Downloads/Create_JSON/aligned_test" #/Users/anguschen/Downloads/Create_JSON/test/test_img
    texture_path = "/Users/anguschen/Downloads/Create_JSON/texture_img_test" #/Users/anguschen/Downloads/Create_JSON/test/test_texture
    output_path = "/Users/anguschen/Downloads/Create_JSON/JSON_file/output.json"
    
    check_img_num(folder_path) #Number of images in the folder : 1136
    check_img_num(texture_path) #Number of images in the folder : 87

    print('Start creating JSON file ...')

    image_data = process_images(folder_path, texture_path)
    save_json(image_data, output_path)

    print('JSON file saved !')