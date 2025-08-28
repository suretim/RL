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