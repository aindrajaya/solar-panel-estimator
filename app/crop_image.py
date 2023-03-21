from PIL import Image
import os

def read_coordinates(coordinates_file):
    coordinates = []
    with open(coordinates_file, 'r') as file:
        for line in file:
            x, y, m, h = map(int, line.strip().split(','))
            coordinates.append((x, y, m, h))
    return coordinates

def crop_image(image_path, coordinates, output_dir):
    # Load the image
    image = Image.open(image_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through the coordinates and crop the image
    for i, (x, y, m, h) in enumerate(coordinates):
        cropped_image = image.crop((x, y, x + m, y + h))

        # Save the cropped image to a separate file
        cropped_image.save(os.path.join(output_dir, f"building_{i+1}.png"))
