import os
from zipfile import ZipFile

import s3fs

from malice import runner
from malice.args import parse_args


def main():
    # Validate arguments
    args = parse_args()
    if args.s3_prefix is None:
        raise ValueError("s3_prefix must be set when running in upload mode.")

    # Run malice
    runner.main()

    # collect all output files
    paths = get_all_file_paths(args.output_dir)

    # Use base of output dir as name of zip file
    base_name = os.path.basename(os.path.normpath(args.output_dir))
    base_name += ".zip"
    zip_name = os.path.join(".", base_name)

    zip_file(zip_name, paths)
    upload_to_s3(base_name, zip_name, args.s3_prefix)

    print("Finished!")
    # TODO: Remove zip file


def upload_to_s3(base_name, zip_name, s3_prefix):
    s3 = s3fs.S3FileSystem(anon=False)
    key_name = s3_prefix + "/" + base_name
    print("Uploading zip to s3://" + key_name)

    with s3.open(key_name, 'wb') as s3_f:
        with open(zip_name, 'rb') as local_f:
            s3_f.write(local_f.read())


def zip_file(zip_name, paths):
    with ZipFile(zip_name, "w") as new_zip:
        print("Zipping these files:")
        for path in paths:
            print(path)
            new_zip.write(path)


# Adapted from https://www.geeksforgeeks.org/working-zip-files-python/
def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths


if __name__ == "__main__":
    main()
