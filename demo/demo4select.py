import cv2
import os
import time
import torch
import argparse
import shutil
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.util.path import mkdir
import asyncio
import numpy as np

image_ext = ['jpg', 'jpeg', 'webp', 'bmp', 'png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
videos_ext = ['.mp4', '.mov', '.avi', '.mkv']

threshold = 0.5


def get_path_list():
    root_path = "/data1/wl/SUN397"

    ima_list = []
    for root, dirs, files in os.walk(root_path):
        for i in files:
            if i.split(sep=".")[-1] in image_ext:
                img_path = os.path.join(root, i)
                ima_list.append(img_path)
    print(len(ima_list))
    return ima_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='image', help='demo type, eg. image, video and webcam')
    parser.add_argument('--config', default="../models/nanodet_ebike_1.0x_v1.6.0_restart.yml", help='model config file path')
    parser.add_argument('--model', default="../models/model_last_v1.6.0_restart_290.ckpt", help='model file path')
    parser.add_argument('--path', default='/data1/wl/videos/Onlyperson', help='path to images or video')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    parser.add_argument('--save_result', default="/data1/wl/detect/v1.6.2", action='store_false',
                        help='whether to save the inference result of image/video')
    args = parser.parse_args()

    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cpu'):
        self.cfg = cfg
        model = build_model(cfg.model)
        self.device = device
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == 'RepVGG':
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({'deploy': True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(self.device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {'id': 0}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def indefenceBatch(self, imgs, batch_size):
        metas, results = [], []
        tmp = 0
        for img in imgs:
            img_info = {'id': 0}
            if isinstance(img, str):
                img_info['file_name'] = os.path.basename(img)
                img = cv2.imread(img)
            else:
                img_info['file_name'] = None

            height, width = img.shape[:2]
            img_info['height'] = height
            img_info['width'] = width
            meta = dict(img_info=img_info,
                        raw_img=img,
                        img=img)
            meta = self.pipeline(meta, self.cfg.data.val.input_size)

            results.append(meta['img'].transpose(2, 0, 1))
            meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0)
            metas.append(meta)
        results = torch.Tensor(results)
        print(results.shape)
        with torch.no_grad():
            results = self.model.inferenceN(results, metas)

        return metas, results

    def visualize(self, dets, meta, class_names, score_thres, show=True, draw=True, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=show)
        print('viz time: {:.3f}s'.format(time.time() - time1))
        return result_img

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def get_video_list(path):
    video_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in videos_ext:
                video_names.append(apath)
    return video_names


def main():
    args = parse_args()
    local_rank = 0
    global threshold
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device='cpu')
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == 'image':
        files = get_path_list()

        files.sort()

        threshold = 0.85
        for image_name in files:
            try:
                meta, res = predictor.inference(image_name)
                result_image = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False, draw=False)

                #
                size_flag = False
                count_ebike = 0
                count_bike = 0
                #
                ebike_flag = False
                bike_flag = False
                # Tip:
                print(res)
                # Tip:
                if count_ebike + count_bike > 1:
                    size_flag = True
                else:
                    size_flag = False

                if args.save_result:
                    save_folder = os.path.join(args.save_result, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
                    mkdir(local_rank, save_folder)
                    mkdir(local_rank, os.path.join(save_folder, "multi_error"))
                    mkdir(local_rank, os.path.join(save_folder, "ebike_error"))
                    mkdir(local_rank, os.path.join(save_folder, "bike_error"))
                    mkdir(local_rank, os.path.join(save_folder, "normal"))
                    #
                    if size_flag:
                        save_file_name = os.path.join(save_folder, "multi_error",
                                                      "{}_{}".format("multi_error", os.path.basename(image_name)))
                    elif ebike_flag:
                        save_file_name = os.path.join(save_folder, "ebike_error",
                                                      "{}_{}".format("ebike_error", os.path.basename(image_name)))
                    elif bike_flag:
                        save_file_name = os.path.join(save_folder, "bike_error",
                                                      "{}_{}".format("bike_error", os.path.basename(image_name)))
                    else:
                        save_file_name = os.path.join(save_folder, "normal",
                                                      "{}_{}".format("normal", os.path.basename(image_name)))
                    #
                # print(save_file_name)
                cv2.imwrite(save_file_name, result_image)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break
            except:
                mkdir(0, '/data1/wl/not_read')
                shutil.copyfile(image_name, '/data1/wl/not_read/' + os.path.basename(image_name))
    #
    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        i = 0
        threshold = 0.8
        while True:
            ret_val, frame = cap.read()
            i += 1
            if ret_val and i % 20 == 0:
                meta, res = predictor.inference(frame)
                result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=True, draw=True)
                #
                #
                size_flag = False
                count_ebike = 0
                count_bike = 0
                #
                ebike_flag = False
                bike_flag = False
                # Tip:
                for r in res[0][0]:
                    if r[4] > threshold:
                        ebike_flag = True
                        count_ebike += 1
                        break
                    else:
                        ebike_flag = False

                # Tip:
                for r in res[0][1]:
                    if r[4] > threshold:
                        bike_flag = True
                        count_bike += 1
                        break
                    else:
                        bike_flag = False
                # Tip:
                if count_ebike + count_bike > 1:
                    size_flag = True
                else:
                    size_flag = False
                #
                image_name = "{}.jpg".format(i)
                #
                if size_flag:
                    save_file_name = os.path.join(save_folder,
                                                  "{}_{}".format("multi_error", os.path.basename(image_name)))
                    cv2.imwrite(save_file_name, result_frame)

                elif ebike_flag:
                    save_file_name = os.path.join(save_folder,
                                                  "{}_{}".format("ebike_error", os.path.basename(image_name)))
                    cv2.imwrite(save_file_name, result_frame)

                elif bike_flag:
                    save_file_name = os.path.join(save_folder,
                                                  "{}_{}".format("bike_error", os.path.basename(image_name)))
                    cv2.imwrite(save_file_name, result_frame)

                else:
                    # save_file_name = os.path.join(save_folder, "{}_{}".format("normal", os.path.basename(image_name)))
                    pass

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break

    elif args.demo == 'videos':
        if os.path.isdir(args.path):
            files = get_video_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for file in files:
            cap = cv2.VideoCapture(file)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            save_folder = os.path.join(args.save_result, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            mkdir(local_rank, save_folder)
            mkdir(local_rank, os.path.join(save_folder, "multi_error"))
            mkdir(local_rank, os.path.join(save_folder, "ebike_error"))
            mkdir(local_rank, os.path.join(save_folder, "bike_error"))
            mkdir(local_rank, os.path.join(save_folder, "normal"))
            i = 0
            count = 0
            now_time = time.time()
            while True:
                ret_val, frame = cap.read()
                if ret_val:
                    if count % 8 == 0:
                        meta, res = predictor.inference(frame)

                        """
                        print()
                        image_name = "{}_{}.jpg".format(file.split(sep='\\')[-1], i)
                        for r in res[0][0]:
                            if r[4] > threshold:
                                i += 1
                                ebike_flag = True
                                # image_res = frame[int(r[1]): int(r[3]), int(r[0]): int(r[2])]
                                save_file_name = os.path.join(save_folder, "ebike_error",
                                                                  "{}_{}".format("ebike_error", os.path.basename(image_name)))
                                print(save_file_name)
                                try:
                                    cv2.imwrite(save_file_name, frame)
                                except:
                                    pass

                            else:
                                ebike_flag = False

                        """
                        result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False)

                        size_flag = False
                        count_bike = 0
                        count_ebike = 0
                        #
                        ebike_flag = False
                        bike_flag = False
                        # Tip:
                        for r in res[0][0]:
                            if r[4] > threshold:
                                ebike_flag = True
                                count_ebike += 1
                                break
                            else:
                                ebike_flag = False

                        # Tip:
                        for r in res[0][1]:
                            if r[4] > threshold:
                                bike_flag = True
                                count_bike += 1
                                break
                            else:
                                bike_flag = False
                        # Tip:
                        if count_ebike + count_bike > 1:
                            size_flag = True
                        else:
                            size_flag = False
                        image_name = "{}_{}.jpg".format(file.split(sep='\\')[-1], i)
                        if size_flag:
                            save_file_name = os.path.join(save_folder, "multi_error",
                                                          "{}_{}".format("multi_error", os.path.basename(image_name)))
                        elif ebike_flag:
                            save_file_name = os.path.join(save_folder, "ebike_error",
                                                          "{}_{}".format("ebike_error", os.path.basename(image_name)))
                        elif bike_flag:
                            save_file_name = os.path.join(save_folder, "bike_error",
                                                          "{}_{}".format("bike_error", os.path.basename(image_name)))
                        else:
                            save_file_name = os.path.join(save_folder, "normal",
                                                          "{}_{}".format("normal", os.path.basename(image_name)))
                        i += 1
                        cv2.imwrite(save_file_name, result_frame)

                        ch = cv2.waitKey(1)
                        if ch == 27 or ch == ord('q') or ch == ord('Q'):
                            break
                    count += 1
                else:
                    break

            end_time = time.time()
            print("time:", end_time-now_time)

    elif args.demo == "infer_videos":
        files = get_video_list(args.path)
        for file in files:
            cap = cv2.VideoCapture(file)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            save_folder = os.path.join(args.save_result, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            mkdir(local_rank, save_folder)
            mkdir(local_rank, os.path.join(save_folder, "multi_error"))
            mkdir(local_rank, os.path.join(save_folder, "ebike_error"))
            mkdir(local_rank, os.path.join(save_folder, "bike_error"))
            mkdir(local_rank, os.path.join(save_folder, "normal"))

            count = 0
            i = 0
            batch_size = 15
            imgs = []
            now_time = time.time()
            while True:
                ret_val, frame = cap.read()
                if ret_val:
                    if count % 8 == 0:
                        imgs.append(frame)
                    count += 1
                else:
                    metas, results = predictor.indefenceBatch(imgs, batch_size)
                    # print(metas)
                    print()
                    for meta, result in zip(metas, results):
                        # print(result)
                        result_frame = predictor.visualize(result[0], meta, cfg.class_names, threshold, show=False)
                        size_flag = False
                        count_bike = 0
                        count_ebike = 0
                        #
                        ebike_flag = False
                        bike_flag = False
                        # Tip:
                        for r in result[0][0]:
                            if r[4] > threshold:
                                ebike_flag = True
                                count_ebike += 1
                                break
                            else:
                                ebike_flag = False

                        # Tip:
                        for r in result[0][1]:
                            if r[4] > threshold:
                                bike_flag = True
                                count_bike += 1
                                break
                            else:
                                bike_flag = False
                        # Tip:
                        if count_ebike + count_bike > 1:
                            size_flag = True
                        else:
                            size_flag = False
                        image_name = "{}_{}.jpg".format(file.split(sep='\\')[-1], i)
                        if size_flag:
                            save_file_name = os.path.join(save_folder, "multi_error",
                                                          "{}_{}".format("multi_error", os.path.basename(image_name)))
                        elif ebike_flag:
                            save_file_name = os.path.join(save_folder, "ebike_error",
                                                          "{}_{}".format("ebike_error", os.path.basename(image_name)))
                        elif bike_flag:
                            save_file_name = os.path.join(save_folder, "bike_error",
                                                          "{}_{}".format("bike_error", os.path.basename(image_name)))
                        else:
                            save_file_name = os.path.join(save_folder, "normal",
                                                          "{}_{}".format("normal", os.path.basename(image_name)))
                        i += 1
                        cv2.imwrite(save_file_name, result_frame)

                        ch = cv2.waitKey(1)
                        if ch == 27 or ch == ord('q') or ch == ord('Q'):
                            break
                    imgs = []  # 推理完制空
                    break

                if len(imgs) == batch_size:
                    print(">>> ", count)
                    metas, results = predictor.indefenceBatch(imgs, batch_size)
                    # print(len(metas)
                    # print(metas)
                    print(results)
                    print()
                    for meta, result in zip(metas, results):
                        # print(result)
                        result_frame = predictor.visualize(result[0], meta, cfg.class_names, threshold, show=False)
                        size_flag = False
                        count_bike = 0
                        count_ebike = 0
                        #
                        ebike_flag = False
                        bike_flag = False
                        # Tip:
                        for r in result[0][0]:
                            if r[4] > threshold:
                                ebike_flag = True
                                count_ebike += 1
                                break
                            else:
                                ebike_flag = False

                        # Tip:
                        for r in result[0][1]:
                            if r[4] > threshold:
                                bike_flag = True
                                count_bike += 1
                                break
                            else:
                                bike_flag = False
                        # Tip:
                        if count_ebike + count_bike > 1:
                            size_flag = True
                        else:
                            size_flag = False
                        image_name = "{}_{}.jpg".format(file.split(sep='\\')[-1], i)
                        if size_flag:
                            save_file_name = os.path.join(save_folder, "multi_error",
                                                          "{}_{}".format("multi_error", os.path.basename(image_name)))
                        elif ebike_flag:
                            save_file_name = os.path.join(save_folder, "ebike_error",
                                                          "{}_{}".format("ebike_error", os.path.basename(image_name)))
                        elif bike_flag:
                            save_file_name = os.path.join(save_folder, "bike_error",
                                                          "{}_{}".format("bike_error", os.path.basename(image_name)))
                        else:
                            save_file_name = os.path.join(save_folder, "normal",
                                                          "{}_{}".format("normal", os.path.basename(image_name)))
                        i += 1
                        cv2.imwrite(save_file_name, result_frame)

                        ch = cv2.waitKey(1)
                        if ch == 27 or ch == ord('q') or ch == ord('Q'):
                            break
                    imgs = []  # 推理完制空
            end_time = time.time()

            print("time:", end_time-now_time)


if __name__ == '__main__':
    main()
