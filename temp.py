import tensorflow as tf

# Check if TensorFlow is able to use the GPU
if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("Available devices:")
print(tf.config.list_physical_devices())
