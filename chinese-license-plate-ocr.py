import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
import traceback
from os.path import splitext, basename
from glob import glob
from src.keras_ocr_utils import LPR

if __name__ == '__main__':
    try:
        input_dir = sys.argv[1]
        output_dir = input_dir
        imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

        ocrmodel = LPR(sys.argv[2])
        # res_set = []
        print('Performing OCR...')
        for i, img_path in enumerate(imgs_paths):
            print('\tScanning %s' % img_path)
            bname = basename(splitext(img_path)[0])
            res, confidence = ocrmodel.recognizeOne(img_path)
            # res_set.append([res, confidence])
            if confidence > 0.5:
                with open('%s/%s_str.txt' % (output_dir, bname), 'w', encoding='utf-8') as f:
                    f.write(res + '\n')
                print('\t\tLP: %s' % res)
            else:
                print('license plate confidence is lower than 0.5!')

    except:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
