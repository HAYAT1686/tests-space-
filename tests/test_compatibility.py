import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import tensorflow as tf
import os
from models.simple_model import create_simple_model
from models.nnsmith_model import create_nnsmith_model

class TestCompatibility(unittest.TestCase):
    
    def setUp(self):
        input_shape = 32  # Example input shape
        num_classes = 10  # Example number of classes
        self.simple_model = create_simple_model()
        self.nnsmith_model = create_nnsmith_model(input_shape, num_classes)  # Pass the required arguments
    
    def test_cpu_compatibility(self):
        with tf.device('/CPU:0'):
            x = tf.random.uniform((1, 32))
            @tf.function(jit_compile=True)
            def compiled_model(x):
                return self.simple_model(x)
            compiled_output = compiled_model(x)
            self.assertIsNotNone(compiled_output)
            
            x = tf.random.uniform((1, 32))  # Assuming the input shape is compatible
            @tf.function(jit_compile=True)
            def compiled_model_nnsmith(x):
                return self.nnsmith_model(x)
            compiled_output = compiled_model_nnsmith(x)
            self.assertIsNotNone(compiled_output)
    
    def test_gpu_compatibility(self):
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                x = tf.random.uniform((1, 32))
                @tf.function(jit_compile=True)
                def compiled_model(x):
                    return self.simple_model(x)
                compiled_output = compiled_model(x)
                self.assertIsNotNone(compiled_output)
                
                x = tf.random.uniform((1, 32))  # Assuming the input shape is compatible
                @tf.function(jit_compile=True)
                def compiled_model_nnsmith(x):
                    return self.nnsmith_model(x)
                compiled_output = compiled_model_nnsmith(x)
                self.assertIsNotNone(compiled_output)
    
    def test_tpu_compatibility(self):
        if 'COLAB_TPU_ADDR' in os.environ:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            with strategy.scope():
                x = tf.random.uniform((1, 32))
                @tf.function(jit_compile=True)
                def compiled_model(x):
                    return self.simple_model(x)
                compiled_output = compiled_model(x)
                self.assertIsNotNone(compiled_output)
                
                x = tf.random.uniform((1, 32))  # Assuming the input shape is compatible
                @tf.function(jit_compile=True)
                def compiled_model_nnsmith(x):
                    return self.nnsmith_model(x)
                compiled_output = compiled_model_nnsmith(x)
                self.assertIsNotNone(compiled_output)

if __name__ == '__main__':
    unittest.main()
