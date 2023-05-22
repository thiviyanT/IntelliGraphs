#!/bin/bash

# Create the .data directory if it doesn't exist
mkdir -p .data

# Define the URLs and filenames of the zip files
urls=(
    "https://www.dropbox.com/s/kp1xp2rbieib4gl/syn-paths.zip?dl=1"
    "https://www.dropbox.com/s/wgm2yr7h8dhcj52/syn-tipr.zip?dl=1"
    "https://www.dropbox.com/s/yx7vrvsxme53xce/syn-types.zip?dl=1"
    "https://www.dropbox.com/s/37etzy2pkix84o8/wd-articles.zip?dl=1"
    "https://www.dropbox.com/s/gavyilqy1kb750f/wd-movies.zip?dl=1"
)
filenames=(
    "syn-paths.zip"
    "syn-tipr.zip"
    "syn-types.zip"
    "wd-articles.zip"
    "wd-movies.zip"
)

# Download and unzip each zip file
for ((i=0; i<${#urls[@]}; i++)); do
    url="${urls[i]}"
    filename="${filenames[i]}"

    # Download the zip file
    echo "Downloading $filename..."
    curl -L -o ".data/$filename" "$url"

    # Unzip the file
    echo "Unzipping $filename..."
    unzip ".data/$filename" -d ".data/"

    # Remove the zip file
    rm ".data/$filename"
done

echo "All zip files have been downloaded and unzipped."
