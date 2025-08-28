from PIL import Image, ImageDraw, ImageFilter
import numpy as np

img_size = (64, 64)

def generate_healthy_image():
    img = Image.new("RGB", img_size, color=(34, 139, 34))  # 绿色叶片
    draw = ImageDraw.Draw(img)
    for _ in range(np.random.randint(5, 20)):
        x, y = np.random.randint(0, 64, 2)
        r = np.random.randint(2, 5)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(30, 130, 30))
    return img

def generate_disease_image():
    img = Image.new("RGB", img_size, color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    for _ in range(np.random.randint(5, 10)):
        x, y = np.random.randint(0, 64, 2)
        r = np.random.randint(5, 15)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(150, 75, 0))
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img

# 选择生成哪种图片
mode = "disease"  # 选 "healthy" 或 "disease"

if mode == "healthy":
    new_leaf = generate_healthy_image()
else:
    new_leaf = generate_disease_image()

# 保存图片
new_leaf.save("new_leaf.jpg")
print("✅ 已生成 new_leaf.jpg")
