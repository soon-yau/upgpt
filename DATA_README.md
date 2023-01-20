# Prepare Data
1. Download img_highres_seg.zip and unzip to ./datasets/deepfashion_inshop/
2. Run script to convert segmentation labels to match multimodal dataset
'''
python scripts/convert_segm.py
python scripts/segment.py
python scripts/resize_highres.py
'''
3. Merge segmentation files
4. Extract clothing styels
 