# Download cityscapes
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=romil.bhardwaj@berkeley.edu&password=Qweasdzxc#12&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

unzip gtFine.zip
unzip leftImg8bit_trainvaltest.zip

# Dont remember why but our dataset moves some train images for test
rm -rf leftImg8bit/test/*
mv leftImg8bit/train/krefeld leftImg8bit/train/monchengladbach leftImg8bit/train/stuttgart leftImg8bit/train/zurich -t leftImg8bit/test


# copy sample lists
mkdir sample_lists
# Run locally rsync -ariz "/media/romilb/NEW VOLUME/cityscapes/dataset/sample_lists" c69:/data/romilb/datasets/cityscapes/

# Run training
python __main__.py --dataset cityscapes --root "/data/romilb/datasets/cityscapes" --list-name "sample_lists/citywise/aachen" -tdd timeseries_sampleincr --sample-incremental -e 1 --batch-size 128 --use-train-for-test