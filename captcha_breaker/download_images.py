import argparse
import requests
import time
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
ap.add_argument("-n", "--num-images", type=int, default=500, help="path to output directory of images")
args = vars(ap.parse_args())

url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 200

for i in range(100, args["num_images"]):
    try:
        r = requests.get(url, timeout=5)
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()
        print("[INFO] downloaded: {}".format(p))
        total +=1

    except:
        print("[INFO error downloading image....]")

    time.sleep(0.1)
