import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
#pip install Pillow

# 参数设置
train_dir = "../../../../dataset/sprout_y_n_data3/train"
val_dir = "../../../../dataset/sprout_y_n_data3/val"
test_dir = "../../../../dataset/sprout_y_n_data3/test"
categories = ["y", "w", "n"]
img_size = (64, 64)
num_images = 100  # 每类生成100张


def generate_sprouts_image():
    # 淺色背景（模擬培養基或土壤）
    bg_color = (240, 240, 200)  # 淺米色
    img = Image.new("RGB", img_size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # 繪製豆芽（白色細長莖 + 綠色小葉）
    for _ in range(np.random.randint(3, 8)):  # 3-7 根豆芽
        x = np.random.randint(10, 54)  # 避免太靠邊緣
        height = np.random.randint(20, 40)  # 豆芽高度

        # 豆芽莖（白色/淺黃色）
        draw.line((x, 64, x, 64 - height),
                  fill=(220, 220, 180),
                  width=np.random.randint(1, 3))

        # 豆芽葉（頂部的小綠點）
        if np.random.random() > 0.3:  # 70% 概率有葉子
            leaf_size = np.random.randint(2, 5)
            draw.ellipse((x - leaf_size, 64 - height - leaf_size,
                          x + leaf_size, 64 - height + leaf_size),
                         fill=(100, 200, 100))

    # 添加一些噪點模擬土壤質感
    for _ in range(50):
        x, y = np.random.randint(0, 64, 2)
        draw.point((x, y), fill=(
            240 + np.random.randint(-20, 10),
            240 + np.random.randint(-20, 10),
            200 + np.random.randint(-20, 10)
        ))
    return img


def generate_non_sprouts_image():
    # 隨機選擇一種非豆芽植物（例如雜草、小灌木等）
    choice = np.random.choice(["grass", "weed", "small_plant"])

    img = Image.new("RGB", img_size, color=(240, 240, 200))  # 淺色背景
    draw = ImageDraw.Draw(img)

    if choice == "grass":
        # 模擬雜草（多條細線）
        for _ in range(np.random.randint(5, 15)):
            x = np.random.randint(10, 54)
            height = np.random.randint(10, 30)
            draw.line((x, 64, x + np.random.randint(-5, 5), 64 - height),
                      fill=(0, 100 + np.random.randint(0, 100), 0),
                      width=1)

    elif choice == "weed":
        # 模擬闊葉雜草
        for _ in range(np.random.randint(2, 5)):
            x, y = np.random.randint(15, 50, 2)
            size = np.random.randint(8, 15)
            draw.ellipse((x - size, y - size, x + size, y + size),
                         fill=(50, 120 + np.random.randint(0, 50), 50))

    else:  # small_plant
        # 模擬小型植物（莖+葉）
        stem_x = np.random.randint(20, 44)
        draw.line((stem_x, 64, stem_x, 64 - np.random.randint(15, 30)),
                  fill=(0, 80, 0), width=2)
        for _ in range(np.random.randint(3, 6)):
            leaf_x = stem_x + np.random.randint(-10, 10)
            leaf_y = 64 - np.random.randint(10, 25)
            leaf_size = np.random.randint(3, 8)
            draw.ellipse((leaf_x - leaf_size, leaf_y - leaf_size,
                          leaf_x + leaf_size, leaf_y + leaf_size),
                         fill=(30, 150, 30))

    return img


def generate_noice_image():
    # 純背景（無植物）
    gray_value = np.random.randint(150, 220)
    return Image.new("RGB", img_size, color=(gray_value, gray_value, gray_value))
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
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
for category in tqdm(categories, desc="Generating images"):
    for i in range(num_images):
        if category == "y":
            img = generate_sprouts_image()
        elif category == "n":
            img = generate_non_sprouts_image()
        else:  # "wu" category
            img = generate_wu_image()
        img.save(os.path.join(train_dir, category, f"{category}_{i}.png"))
#for category in tqdm(categories, desc="Generating images"):
    for i in range(num_images):
        if category == "y":
            img = generate_sprouts_image()
        elif category == "n":
            img = generate_non_sprouts_image()
        else:  # "wu" category
            img = generate_wu_image()
        img.save(os.path.join(val_dir, category, f"{category}_{i}.png"))
#for category in tqdm(categories, desc="Generating images"):
    for i in range(num_images):
        if category == "y":
            img = generate_sprouts_image()
        elif category == "n":
            img = generate_non_sprouts_image()
        else:  # "wu" category
            img = generate_wu_image()
        img.save(os.path.join(test_dir, category, f"{category}_{i}.png"))

