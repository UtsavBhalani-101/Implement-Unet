import cv2, os, PIL, shutil
import numpy as np, nibabel as nib
from PIL import Image
from pydicom import dcmread

def nii_to_png_3ch(filename,output_dir,ww = 130,wc = 25):
    os.makedirs(output_dir, exist_ok=True)
        
    ds = nib.load(filename)
        
        
    np_data = np.array(ds.dataobj)
    lower_bound = wc - 0.5*ww
    upper_bound = wc + 0.5*ww

    np_data = np.where(np_data < lower_bound, lower_bound, np_data )
    np_data = np.where(np_data > upper_bound, upper_bound, np_data)

    np_data = 255/(upper_bound - lower_bound)*(np_data - lower_bound)
    np_data = np_data.astype('uint8')

    l = np_data.shape[2] 
        
    for i in range(l):
        path = os.path.join(output_dir,os.path.basename(filename).split('.')[0]+"_"+str(i)+'.png')
        prev = np.flip(np.swapaxes(np_data[:,:,max(i-1,0)],0,1))
        curr = np.flip(np.swapaxes(np_data[:,:,i],0,1))
        pro = np.flip(np.swapaxes(np_data[:,:,min(i+1,l-1)],0,1))
        image = np.stack([prev,curr,pro],axis= 2)
            
        im = Image.fromarray(image)
        im.save(path) 
        


def dcm_to_np(filename):
    ds = dcmread(filename)

    try:
        slope=ds.RescaleSlope
        intercept=ds.RescaleIntercept
    except:
        slope =1 
        intercept =0

    np_data = ds.pixel_array*slope + intercept

    return np_data

def dcm_to_png_3ch(dcm_dir, output_dir, ww=130, wc=25):
    """Convert one ordered DICOM study into three-slice PNG inputs.

    This preserves the authors' public API while fixing the original syntax
    and tuple/path errors.  It orders files by InstanceNumber and does not
    load all pixel arrays into memory at once.
    """
    lower_bound = wc - 0.5 * ww
    upper_bound = wc + 0.5 * ww
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [
        os.path.join(dcm_dir, name)
        for name in os.listdir(dcm_dir)
        if name.lower().endswith(".dcm")
    ]
    if not image_paths:
        raise FileNotFoundError(f"no .dcm files found in {dcm_dir}")

    def instance_number(path):
        value = dcmread(path, stop_before_pixels=True).get("InstanceNumber")
        if value is None:
            raise ValueError(f"missing InstanceNumber in {path}")
        return int(value)

    sorted_paths = sorted(image_paths, key=instance_number)
    for i, current_path in enumerate(sorted_paths):
        previous_path = sorted_paths[max(i - 1, 0)]
        next_path = sorted_paths[min(i + 1, len(sorted_paths) - 1)]
        channels = np.stack([
            dcm_to_np(previous_path),
            dcm_to_np(current_path),
            dcm_to_np(next_path),
        ], axis=2)
        channels = np.clip(channels, lower_bound, upper_bound)
        channels = (255 / (upper_bound - lower_bound) *
                    (channels - lower_bound)).astype("uint8")
        output_path = os.path.join(output_dir, os.path.basename(current_path) + ".png")
        Image.fromarray(channels).save(output_path)
