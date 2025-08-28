# 评估测试集
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


# 预测单张图像
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=IMG_SIZE
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    pred = model.predict(img_array)
    pred_class = class_names[tf.argmax(pred, axis=1)[0]]
    confidence = tf.reduce_max(pred).numpy()

    return pred_class, confidence


# 使用示例
image_path = "test_sprout.jpg"
pred_class, confidence = predict_image(image_path)
print(f"Predicted: {pred_class} (Confidence: {confidence * 100:.2f}%)")