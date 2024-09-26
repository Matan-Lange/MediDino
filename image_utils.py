import os
import shutil
import argparse


def copy_images(paths, new_path):
    for path in paths:
        for file in os.listdir(path):
            #write
            if file.endswith('.png') or file.endswith('.jpg'):
                shutil.copy(os.path.join(path, file), new_path)
    return
def main():
    parser = argparse.ArgumentParser(description='Copy images from multiple directories to a new directory.')
    parser.add_argument('directories', nargs='+', help='List of directories containing images')
    parser.add_argument('new_directory', help='Directory to copy images to')

    args = parser.parse_args()

    copy_images(args.directories, args.new_directory)

    print(f'Copied images from {len(args.directories)} directories to {args.new_directory}')


if __name__ == '__main__':
    main()
