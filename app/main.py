from app.crop_image import read_coordinates, crop_image

if __name__ == "__main__":
    # Prompt the user for the image and coordinates files
    image_path = input("Enter the path to the image file: ")
    coordinates_file = input("Enter the path to the coordinates file: ")
    output_dir = input("Enter the name of your output directory: ")

    # Read the coordinates from the coordinates file
    coordinates = read_coordinates(coordinates_file)

    # Crop the image and save the cropped parts
    crop_image(image_path, coordinates, output_dir)
