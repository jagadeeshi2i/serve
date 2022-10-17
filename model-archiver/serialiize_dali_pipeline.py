# The MIT License (MIT)
#
# Copyright (c) 2021 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nvidia.dali as dali
import nvidia.dali.types as types
import os
from nvidia.dali.pipeline import pipeline_def


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="./model_repository/model.dali")
    return parser.parse_args()


@dali.pipeline_def
def pipe():
    jpegs = dali.fn.external_source(dtype=types.UINT8, name="my_source")
    decoded = dali.fn.decoders.image(jpegs, device='mixed')
    resized = dali.fn.resize(decoded, size=[256])
    normalized = dali.fn.crop_mirror_normalize(
            decoded,
            # crop_w = 36,
            # crop_h = 36,
            crop_pos_x = 0.5,
            crop_pos_y = 0.5,
            crop=(224,224),
            mean=[0.485*255, 0.456*255, 0.406*255],
            std=[0.229*255, 0.224*255, 0.225*255]
    )
    return normalized


def main(filename):
    pipe1 = pipe(batch_size=1, num_threads=2, device_id=0, seed = 13)
    pipe.serialize(filename="model1.dali")
    print("Saved {}".format(filename))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    main(args.save)