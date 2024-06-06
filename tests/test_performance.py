import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import tensorflow as tf
import time
from models.simple_model import create_simple_model
from models.nnsmith_model import create_nnsmith_model

class TestPerformance(unittest.TestCase):
    
    def setUp(self):
        input_shape = 32  # Example input shape
        num_classes = 10  # Example number of classes
        self.simple_model = create_simple_model()
        self.nnsmith_model = create_nnsmith_model(input_shape, num_classes)  # Pass the required arguments
    
    def measure_execution_time(self, model, x):
        start_time = time.time()
        _ = model(x)
        return time.time() - start_time
    
    def test_simple_model_performance(self):
        x = tf.random.uniform((1024, 32))
        
        non_compiled_time = self.measure_execution_time(self.simple_model, x)
        
        @tf.function(jit_compile=True)
        def compiled_model(x):
            return self.simple_model(x)
        
        compiled_time = self.measure_execution_time(compiled_model, x)
        
        self.assertLess(compiled_time, non_compiled_time)

    def test_nnsmith_model_performance(self):
        x = tf.random.uniform((1024, 32))  # Assuming the input shape is compatible
        
        non_compiled_time = self.measure_execution_time(self.nnsmith_model, x)
        
        @tf.function(jit_compile=True)
        def compiled_model(x):
            return self.nnsmith_model(x)
        
        compiled_time = self.measure_execution_time(compiled_model, x)
        
        self.assertLess(compiled_time, non_compiled_time)

if __name__ == '__main__':
    unittest.main()
