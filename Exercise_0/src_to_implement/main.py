from generator import ImageGenerator  # Import your class
from pattern import Checker, Circle, Spectrum

# Task 1 - Pattern Testing
checker = Checker(resolution=200, tile_size=20)
print(checker.draw())
checker.show()

circle = Circle(resolution=200, radius=50, position=(100, 100))
circle.draw()
circle.show()

spectrum = Spectrum(resolution=300)
spectrum.draw()
spectrum.show()

# Task 2 - Image Generator Example (You need real image path + JSON label path)
# Example usage (adjust your file paths):
# gen = ImageGenerator('data/', 'data/labels.json', batch_size=4, image_size=(64, 64, 3), rotation=True, mirroring=True, shuffle=True)
# gen.show()

# Initialize the ImageGenerator with your actual data path
gen = ImageGenerator(
    file_path='./data/exercise_data',              # Folder with .npy image files
    label_path='./data/Labels.json', # JSON file with labels
    batch_size=15,                    # Test with batch size 5
    image_size=[32, 32, 3],          # Resize images to this shape
    rotation=False,                   # Enable rotation
    mirroring=True,                  # Enable mirroring
    shuffle=True                     # Enable shuffling per epoch
)

# Test next() method (print shapes)
images, labels = gen.next()
print("Images shape:", images.shape)  # Expected: (5, 32, 32, 3)
print("Labels shape:", labels.shape)  # Expected: (5,)
print("Current Epoch:", gen.current_epoch())

# Visualize the batch
gen.show()
