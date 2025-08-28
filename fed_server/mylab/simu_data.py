import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
#pip install Pillow

# 参数设置
output_dir = "../../../../dataset/general_h_d_data3"
categories = ["h", "wu", "d"]
img_size = (64, 64)
num_images = 100  # 每类生成100张

def generate_healthy_image():
    img = Image.new("RGB", img_size, color=(34, 139, 34))  # 健康绿色
    draw = ImageDraw.Draw(img)
    for _ in range(np.random.randint(5, 20)):
        x, y = np.random.randint(0, 64, 2)
        r = np.random.randint(2, 5)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(30, 130, 30))  # 模拟叶脉
    return img

def generate_disease_image():
    img = Image.new("RGB", img_size, color=(34, 139, 34))  # 同样叶色为底
    draw = ImageDraw.Draw(img)
    for _ in range(np.random.randint(5, 10)):
        x, y = np.random.randint(0, 64, 2)
        r = np.random.randint(5, 15)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(150, 75, 0))  # 棕色病斑
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img


def generate_wu_image():
    """生成无植物的土壤/背景图像"""
    # 随机选择一种生成方式
    choice = np.random.randint(0, 3)

    if choice == 0:  # 纯色土壤
        soil_colors = [(139, 69, 19), (160, 82, 45), (101, 67, 33)]
        img = Image.new("RGB", img_size, color=soil_colors[np.random.randint(0, len(soil_colors))])

    elif choice == 1:  # 带石头的土壤
        img = Image.new("RGB", img_size, color=(139, 69, 19))
        draw = ImageDraw.Draw(img)
        for _ in range(np.random.randint(3, 8)):
            x, y = np.random.randint(0, 64, 2)
            r = np.random.randint(2, 6)
            gray = np.random.randint(100, 200)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(gray, gray, gray))

    else:  # 简单灰色背景
        gray_value = np.random.randint(150, 220)
        img = Image.new("RGB", img_size, color=(gray_value, gray_value, gray_value))

    # 统一添加一些噪点增加真实感
    pixels = img.load()
    for _ in range(np.random.randint(50, 150)):
        x, y = np.random.randint(0, 64, 2)
        if isinstance(pixels[x, y], int):  # 对于某些模式可能是单值
            val = pixels[x, y] + np.random.randint(-20, 20)
            val = max(0, min(255, val))
            pixels[x, y] = val
        else:  # RGB模式
            r, g, b = pixels[x, y]
            pixels[x, y] = (
                max(0, min(255, r + np.random.randint(-20, 20))),
                max(0, min(255, g + np.random.randint(-20, 20))),
                max(0, min(255, b + np.random.randint(-20, 20)))
            )
    return img



# 创建目录
for category in categories:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)
for category in tqdm(categories, desc="Generating images"):
    for i in range(num_images):
        if category == "healthy":
            img = generate_healthy_image()
        elif category == "disease":
            img = generate_disease_image()
        else:  # "wu" category
            img = generate_wu_image()
        img.save(os.path.join(output_dir, category, f"{category}_{i}.png"))

