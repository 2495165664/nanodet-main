# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

from .color import color_aug_and_norm
from .warp import warp_and_resize
from matplotlib import pyplot as plt
import cv2

class Pipeline:
    def __init__(self, cfg, keep_ratio):
        self.warp = functools.partial(
            warp_and_resize, warp_kwargs=cfg, keep_ratio=keep_ratio
        )
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, meta, dst_shape):
        # cvColor_image = cv2.cvtColor(meta['img'], cv2.COLOR_BGR2RGB)
        # plt.imshow(cvColor_image)
        # plt.show()
        meta = self.warp(meta=meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        # for box in meta['gt_bboxes']:
        #     print(list(box))
        #     box = list(box)
        #     cv2.rectangle(meta['img'], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 3)
        # cvColor_image = cv2.cvtColor(meta['img'], cv2.COLOR_BGR2RGB)
        # plt.imshow(cvColor_image)
        # plt.show()
        return meta
