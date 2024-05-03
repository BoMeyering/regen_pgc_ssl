set -e
echo "Downloading S3 datasets"
if [ -f ~/.aws/config ] && [ -f ~/.aws/credentials ]; then
    echo "AWS CREDS EXIST"
    aws s3 cp s3://pgc-datasets/regen-pgc-view ./data/raw/regen-pgc-view --recursive 
    aws s3 cp s3://pgc-datasets/cropandweed-dataset ./data/raw/cropandweed-dataset --recursive
    aws s3 cp s3://pgc-datasets/grass-clover-dataset ./data/raw/grass-clover-dataset --recursive
else
    echo "AWS CREDS DO NOT EXIST"
    echo "Downloading CropAndWeed Dataset"
    cd data/raw
    git clone https://github.com/cropandweed/cropandweed-dataset.git
    cd cropandweed-dataset
    python cnw/setup.py || exit 1
    cd ../
    echo "Downloading grass-clover-dataset 'synthetic_images'"
    if [ ! -d ./grass-clover-dataset ]; then
        mkdir ./grass-clover-dataset
    cd ./grass-clover-dataset
    wget -q --spider https://vision.eng.au.dk/?download=/data/GrassClover/synthetic_images.zip -cO synthetic_images.zip # remove --spider for no dry run
    (unzip synthetic_images.zip && rm -f synthetic_images.zip) || exit 1
    cd ../
    
    echo "Downloading grass-clover-dataset 'developed_images'"
    wget -q --spider https://vision.eng.au.dk/?download=/data/GrassClover/developed_images.zip -cO developed_images.zip # remove --spider for no dry run
    (unzip developed_images.zip && rm -f developed_images.zip) || exit 1
    cd ../

    echo "Downloading Regen-pgc-view"
    # implement later
fi

echo "All files downloaded successfully"
exit 0