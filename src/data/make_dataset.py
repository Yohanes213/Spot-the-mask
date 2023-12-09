import os
import glob
from zipfile import ZipFile


def extract_zip(zip_file_path, extract_to_path):
	with ZipFile(zip_file_path, 'r') as zip_ref:
		zip_ref.extractall(path=extract_to_path)


if __name__ == '__main__':

	current_directory = os.path.dirname(os.path.abspath(__file__))

	# Path to the images.zip file

	zip_file_path = os.path.join(current_directory, os.pardir, os.pardir, 'data', 'raw', 'images.zip')

	# Path to the directory where we want to extract the contents

	extract_to_path = os.path.join(current_directory, os.pardir, os.pardir, 'data' ,'processed')


	extract_zip(zip_file_path, extract_to_path)

	print(f"Extraction complete. Images extracted to {os.path.join(extract_to_path)}")

