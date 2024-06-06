import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import tensorflow as tf
from models.simple_model import create_simple_model
from models.nnsmith_model import create_nnsmith_model

class TestEdgeCases(unittest.TestCase):
    
    def setUp(self):
        input_shape = 32  # Example input shape
        num_classes = 10  # Example number of classes
        self.simple_model = create_simple_model()
        self.nnsmith_model = create_nnsmith_model(input_shape, num_classes)  # Pass the required arguments

    def test_empty_input(self):
        x = tf.constant([], dtype=tf.float32)
        with self.assertRaises(ValueError):
            @tf.function(jit_compile=True)
            def compiled_model(x):
                return self.simple_model(x)
            compiled_model(x)
        
        with self.assertRaises(ValueError):
            @tf.function(jit_compile=True)
            def compiled_model_nnsmith(x):
                return self.nnsmith_model(x)
            compiled_model_nnsmith(x)

    def test_large_input(self):
        x = tf.random.uniform((100000, 32))
        @tf.function(jit_compile=True)
        def compiled_model(x):
            return self.simple_model(x)
        compiled_output = compiled_model(x)
        self.assertIsNotNone(compiled_output)
        
        x = tf.random.uniform((100000, 32))  # Assuming the input shape is compatible
        @tf.function(jit_compile=True)
        def compiled_model_nnsmith(x):
            return self.nnsmith_model(x)
        compiled_output = compiled_model_nnsmith(x)
        self.assertIsNotNone(compiled_output)

    def test_uncommon_shapes(self):
        x = tf.random.uniform((1, 1, 32))
        @tf.function(jit_compile=True)
        def compiled_model(x):
            return self.simple_model(x)
        compiled_output = compiled_model(x)
        self.assertIsNotNone(compiled_output)
        
        x = tf.random.uniform((1, 1, 32))  # Assuming the input shape is compatible
        @tf.function(jit_compile=True)
        def compiled_model_nnsmith(x):
            return self.nnsmith_model(x)
        compiled_output = compiled_model_nnsmith(x)
        self.assertIsNotNone(compiled_output)

if __name__ == '__main__':
    unittest.main()
