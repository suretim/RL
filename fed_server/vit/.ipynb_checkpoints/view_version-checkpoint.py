import tensorflow as tf



#pip install tensorflow==2.12.0
#pip uninstall tensorflow keras -y
#pip cache purge
#pip install --user tensorflow==2.12.0 keras==2.12.0
#python -m venv tf_env
#source tf_env/bin/activate  # Linux/Mac
#tf_env\Scripts\activate     # Windows
#pip install tensorflow==2.12.0 tensorflow-hub==0.13.0 keras==2.12.0 numpy==1.23.5 tensorboard==2.12.0
#pip install tensorflow==2.12.0 keras==2.12.0 numpy==1.23.5 tensorboard==2.12.0 --force-reinstall
#Remove-Item -Recurse -Force C:\Users\YourName\miniconda3\envs\tf_env
print(tf.__version__)  # 应显示您安装的版本
print(tf.keras.__version__)  # 对应版本应自动匹配

print(f"TF: {tf.__version__}, Keras: {tf.keras.__version__}")
print(f"NumPy: {tf.__version__}, TensorBoard: {tf.__version__}")
# 测试基本功能
model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
model.compile(optimizer='adam', loss='mse')
print("安装成功！")


# 应输出类似：
# TF: 2.12.0, Keras: 2.12.0
# NumPy: 1.23.5, TensorBoard: 2.12.0
