import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import tensorflow as tf
from models.simple_model import create_simple_model
from models.complex_model import create_complex_model
from models.nnsmith_model import create_nnsmith_model

class TestCorrectness(unittest.TestCase):
    
    def setUp(self):
        input_shape = 32  # Example input shape
        num_classes = 10  # Example number of classes
        self.simple_model = create_simple_model()
        self.complex_model = create_complex_model()
        self.nnsmith_model = create_nnsmith_model(input_shape, num_classes)  # Pass the required arguments
    
    def test_simple_model_correctness(self):
        x = tf.random.uniform((1, 32))
        expected_output = self.simple_model(x)
        
        @tf.function(jit_compile=True)
        def compiled_model(x):
            return self.simple_model(x)
        
        compiled_output = compiled_model(x)
        
        self.assertTrue(tf.reduce_all(tf.equal(expected_output, compiled_output)))

    def test_complex_model_correctness(self):
        x = tf.random.uniform((1, 64))
        expected_output = self.complex_model(x)
        
        @tf.function(jit_compile=True)
        def compiled_model(x):
            return self.complex_model(x)
        
        compiled_output = compiled_model(x)
        
        self.assertTrue(tf.reduce_all(tf.equal(expected_output, compiled_output)))

    def test_nnsmith_model_correctness(self):
        x = tf.random.uniform((1, 32))  # Assuming the input shape is compatible
        expected_output = self.nnsmith_model(x)
        
        @tf.function(jit_compile=True)
        def compiled_model(x):
            return self.nnsmith_model(x)
        
        compiled_output = compiled_model(x)
        
        self.assertTrue(tf.reduce_all(tf.equal(expected_output, compiled_output)))

if __name__ == '__main__':
    unittest.main()
