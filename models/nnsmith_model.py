import tensorflow as tf
from tensorflow.keras import layers, models

# Function to generate a neural network model
def generate_model(input_shape, num_classes):
    # Define the model
    model = models.Sequential()
    
    # Add layers to the model
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Function to create and compile the model
def create_nnsmith_model(input_shape, num_classes):
    # Generate a random neural network
    model = generate_model(input_shape, num_classes)
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Example usage
if __name__ == "__main__":
    input_shape = 100  # Example input shape
    num_classes = 10   # Example number of classes

    model = create_nnsmith_model(input_shape, num_classes)
    model.summary()  # To print the summary of the model
