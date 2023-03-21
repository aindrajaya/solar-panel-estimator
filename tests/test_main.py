import os
from app.crop_image import crop_image

def test_crop_image():
    # Define test input values
    image_path = "test_image.jpg"
    coordinates = [(100, 100, 200, 200), (300, 300, 150, 150), (245, 245, 100, 100)]
    output_dir = "test_output"

    # Call the crop_image function
    crop_image(image_path, coordinates, output_dir)

    # Assert that the output files were created
    assert os.path.exists("test_output/building_1.png")
    assert os.path.exists("test_output/building_2.png")
    assert os.path.exists("test_output/building_3.png")
