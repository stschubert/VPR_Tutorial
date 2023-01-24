#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================
#
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize


def add_frame(img_in, color):
    img = img_in.copy()

    w = int(np.round(0.01*img.shape[1]))

    # pad left-right
    pad_lr = np.tile(np.uint8(color).reshape(1, 1, 3), (img.shape[0], w, 1))
    img = np.concatenate([pad_lr, img, pad_lr], axis=1)

    # pad top-bottom
    pad_tb = np.tile(np.uint8(color).reshape(1, 1, 3), (w, img.shape[1], 1))
    img = np.concatenate([pad_tb, img, pad_tb], axis=0)

    return img


def show(db_imgs, q_imgs, TP, FP, M=None):
    # true positive TP
    idx_tp = np.random.permutation(len(TP))[:1]

    db_tp = db_imgs[int(TP[idx_tp, 0])]
    q_tp = q_imgs[int(TP[idx_tp, 1])]

    if db_tp.shape != q_tp.shape:
        q_tp = resize(q_tp.copy(), db_tp.shape, anti_aliasing=True)
        q_tp = np.uint8(q_tp*255)

    img = add_frame(np.concatenate([db_tp, q_tp], axis=1), [119, 172, 48])

    # false positive FP
    try:
        idx_fp = np.random.permutation(len(FP))[:1]

        db_fp = db_imgs[int(FP[idx_fp, 0])]
        q_fp = q_imgs[int(TP[idx_fp, 1])]

        if db_fp.shape != q_fp.shape:
            q_fp = resize(q_fp.copy(), db_fp.shape, anti_aliasing=True)
            q_fp = np.uint8(q_fp*255)

        img_fp = add_frame(np.concatenate(
            [db_fp, q_fp], axis=1), [162, 20, 47])
        img = np.concatenate([img, img_fp], axis=0)
    except:
        pass

    # concat M
    if M is not None:
        M = resize(M.copy(), (img.shape[0], img.shape[0]))
        M = np.uint8(M.astype('float32')*255)
        M = np.tile(np.expand_dims(M, -1), (1, 1, 3))
        img = np.concatenate([M, img], axis=1)

    # show
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
