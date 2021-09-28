# -*- coding:utf-8 -*-
import cv2
import os
import time
import torch
import argparse
from tqdm import tqdm
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.util.path import mkdir

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
videos_ext = ['.mp4', '.mov', '.avi', '.mkv']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='image', help='demo type, eg. image, video and webcam')
    parser.add_argument('--config', default='../config/nanodet_v2.0x_person_vMobilNetv3.yml',
                        help='model config file path')
    parser.add_argument('--model', default='./model_best_v1.5.ckpt', help='model file path')
    parser.add_argument('--path', default='/data1/wl/detect_data/bg/UCF-QNRF_ECCV18/Test',
                        help='path to images or video')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    parser.add_argument('--save_result', action='store_false',
                        help='whether to save the inference result of image/video')
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == 'RepVGG':
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({'deploy': True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
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
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, show=True, draw=True, crop=False, xml=False, wait=0,
                  filename=''):
        time1 = time.time()
        result_img = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=show,
                                                 draw=draw, crop=crop, xml=xml, filename=filename)
        # print('viz time: {:.3f}s'.format(time.time()-time1))
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
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device='cuda:0')
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == 'image':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()

        threshold = 0.75
        for image_name in tqdm(files):
            meta, res = predictor.inference(image_name)
            result_image = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False, draw=True)

            #
            size_flag = False
            count_ebike = 0
            max_score_ebike = 0.
            count_bike = 0
            max_score_bike = 0.
            #
            ebike_flag = False
            bike_flag = False
            #
            # Tip: Ebike
            for r in res[0][0]:
                if r[4] > threshold:
                    if max_score_ebike < r[4]:
                        max_score_ebike = r[4]
                    count_ebike += 1
                    break
            if max_score_ebike > threshold:
                ebike_flag = True
            else:
                ebike_flag = False
            #
            # Tip: Bike
            for r in res[0][1]:
                if r[4] > threshold:
                    if max_score_bike < r[4]:
                        max_score_bike = r[4]
                    count_bike += 1
                    break
            if max_score_bike > threshold:
                bike_flag = True
            else:
                bike_flag = False
            #

            if count_ebike + count_bike > 1:
                size_flag = True
            else:
                size_flag = False

            if args.save_result:
                save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
                mkdir(local_rank, save_folder)
                #
                # if size_flag:
                #     save_file_name = os.path.join(save_folder, "{}_{}".format("multi_error",os.path.basename(image_name)))
                # elif ebike_flag:
                #     save_file_name = os.path.join(save_folder, "{}_{}".format("ebike_error",os.path.basename(image_name)))
                # elif bike_flag:
                #     save_file_name = os.path.join(save_folder, "{}_{}".format("bike_error",os.path.basename(image_name)))
                # else:
                #     save_file_name = os.path.join(save_folder, "{}_{}".format("normal", os.path.basename(image_name)))
                #
                # Tip:
                save_file_name = os.path.join(save_folder, "{}".format(os.path.basename(image_name)))
                #
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    #
    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        # save_path = os.path.join(save_folder, args.path.split('/')[-1]) if args.demo == 'video' else os.path.join(save_folder, 'camera.mp4')
        # print(f'save_path is {save_path}')
        # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
        i = 0
        threshold = 0.8
        while True:
            ret_val, frame = cap.read()
            if ret_val:  # and i % 1 == 0:
                meta, res = predictor.inference(frame)
                result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=True, draw=True)
                #
                #
                size_flag = False
                count_ebike = 0
                max_score_ebike = 0.
                count_bike = 0
                max_score_bike = 0.
                #
                ebike_flag = False
                bike_flag = False
                #
                # Tip: Ebike
                for r in res[0][0]:
                    if r[4] > threshold:
                        if max_score_ebike < r[4]:
                            max_score_ebike = r[4]
                        count_ebike += 1
                        break
                if max_score_ebike > threshold:
                    ebike_flag = True
                else:
                    ebike_flag = False
                #
                # Tip: Bike
                for r in res[0][1]:
                    if r[4] > threshold:
                        if max_score_bike < r[4]:
                            max_score_bike = r[4]
                        count_bike += 1
                        break
                if max_score_bike > threshold:
                    bike_flag = True
                else:
                    bike_flag = False
                #
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

                # vid_writer.write(result_frame)

                i += 1
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break
            else:
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
            save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            mkdir(local_rank, save_folder)
            # save_path = os.path.join(save_folder, args.path.split('/')[-1]) if args.demo == 'video' else os.path.join(save_folder, 'camera.mp4')
            # print(f'save_path is {save_path}')
            # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
            i = 0
            threshold = 0.8
            print(file)
            while True:
                ret_val, frame = cap.read()
                if ret_val:
                    if i % 3 == 0:
                        meta, res = predictor.inference(frame)
                        result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=True,
                                                           draw=True)
                        #
                        #
                        size_flag = False
                        count_ebike = 0
                        max_score_ebike = 0.
                        count_bike = 0
                        max_score_bike = 0.
                        #
                        ebike_flag = False
                        bike_flag = False
                        #
                        # Tip: Ebike
                        for r in res[0][0]:
                            if r[4] > threshold:
                                if max_score_ebike < r[4]:
                                    max_score_ebike = r[4]
                                count_ebike += 1
                                break
                        if max_score_ebike > threshold:
                            ebike_flag = True
                        else:
                            ebike_flag = False
                        #
                        # Tip: Bike
                        for r in res[0][1]:
                            if r[4] > threshold:
                                if max_score_bike < r[4]:
                                    max_score_bike = r[4]
                                count_bike += 1
                                break
                        if max_score_bike > threshold:
                            bike_flag = True
                        else:
                            bike_flag = False
                        #
                        # Tip:
                        if count_ebike + count_bike > 1:
                            size_flag = True
                        else:
                            size_flag = False
                        #
                        image_name = "{}_{}.jpg".format(file.split(sep='\\')[-1], i)
                        #
                        if size_flag:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{}".format("multi_error", os.path.basename(image_name)))
                        elif ebike_flag:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{:.2}_{}".format("ebike_error", max_score_ebike,
                                                                               os.path.basename(image_name)))
                        elif bike_flag:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{:.2}_{}".format("bike_error", max_score_bike,
                                                                               os.path.basename(image_name)))
                        else:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{}".format("normal", os.path.basename(image_name)))
                        i += 1
                        cv2.imwrite(save_file_name, result_frame)

                        ch = cv2.waitKey(1)
                        if ch == 27 or ch == ord('q') or ch == ord('Q'):
                            break
                    else:
                        i += 1
                else:
                    break
    elif args.demo == 'imgs_savexml':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        mkdir(local_rank, os.path.join(save_folder, "origin"))
        threshold = 0.55
        for image_name in tqdm(files):

            img_ = cv2.imread(image_name)
            raw_image = None
            raw_image = img_.copy()

            meta, res = predictor.inference(image_name)
            #
            save_origin_file_name = os.path.join(save_folder, "origin", "{}".format(os.path.basename(image_name)))
            #
            # print(save_origin_file_name)
            result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False,
                                               draw=True, xml=True, filename=save_origin_file_name.split('.jpg')[0])
            #
            size_flag = False
            #
            count_ebike = 0
            max_score_ebike = 0.
            count_bike = 0
            max_score_bike = 0.
            #
            ebike_flag = False
            bike_flag = False
            #
            # Tip: Ebike
            for r in res[0][0]:
                if r[4] > threshold:
                    if max_score_ebike < r[4]:
                        max_score_ebike = r[4]
                    count_ebike += 1
                    break
            if max_score_ebike > threshold:
                ebike_flag = True
            else:
                ebike_flag = False
            #
            # Tip: Bike
            for r in res[0][1]:
                if r[4] > threshold:
                    if max_score_bike < r[4]:
                        max_score_bike = r[4]
                    count_bike += 1
                    break
            if max_score_bike > threshold:
                bike_flag = True
            else:
                bike_flag = False
            #
            # Tip:
            if count_ebike + count_bike > 1:
                size_flag = True
            else:
                size_flag = False
            #

            #

            if size_flag:
                save_file_name = os.path.join(save_folder,
                                              "{}_{}".format("multiError", os.path.basename(image_name)))
                cv2.imwrite(save_file_name, result_frame)

                # cv2.imshow("res:{}".format(args.model), result_frame)
                #
                cv2.imwrite(save_origin_file_name, raw_image)


            elif ebike_flag:
                save_file_name = os.path.join(save_folder,
                                              "{}_{}".format("ebikeError", os.path.basename(image_name)))
                cv2.imwrite(save_file_name, result_frame)

                # cv2.imshow("res:{}".format(args.model), result_frame)
                #
                cv2.imwrite(save_origin_file_name, raw_image)


            elif bike_flag:
                save_file_name = os.path.join(save_folder,
                                              "{}_{}".format("bikeError", os.path.basename(image_name)))
                cv2.imwrite(save_file_name, result_frame)

                # cv2.imshow("res:{}".format(args.model), result_frame)
                #
                cv2.imwrite(save_origin_file_name, raw_image)

            else:
                # save_file_name = os.path.join("/media/tang/01D74F390B0BECA0/test", "{}_{}_{}_{}".format(c, "normal", timestamp, os.path.basename(image_name)))
                # cv2.imwrite(save_file_name, cv2.resize(raw_frame, (640, 640)))
                pass

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    elif args.demo == 'big_video_savexml':
        cap = cv2.VideoCapture(args.path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        mkdir(local_rank, os.path.join(save_folder, "origin"))

        i = 0
        threshold = 0.3
        #
        patch_w, patch_h = 5000, 5000
        patch_pos = [[0, 0]]

        while True:
            ret_val, raw_big_frame = cap.read()
            #
            if ret_val:
                if i % 5 == 0:
                    for c, crop_pos in enumerate(patch_pos):
                        frame = raw_big_frame[crop_pos[1]:crop_pos[1] + patch_h,
                                crop_pos[0]:crop_pos[0] + patch_w].copy()
                        #
                        raw_frame = None
                        raw_frame = frame.copy()
                        # cv2.imshow("raw{}".format(c), raw_frame)
                        #
                        meta, res = predictor.inference(frame)

                        #
                        image_name = "{}.jpg".format(i)
                        timestamp = int(time.time())
                        save_origin_file_name = os.path.join(save_folder, "origin",
                                                             "{}_{}_{}".format(
                                                                 timestamp, c,
                                                                 os.path.basename(
                                                                     image_name)))
                        #
                        # print(save_origin_file_name)
                        result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False,
                                                           draw=True, filename=save_origin_file_name.split('.jpg')[0],
                                                           xml=True)
                        cv2.imshow("{}_crop".format(c), result_frame)
                        #
                        #
                        size_flag = False
                        #
                        count_ebike = 0
                        max_score_ebike = 0.
                        count_bike = 0
                        max_score_bike = 0.
                        #
                        ebike_flag = False
                        bike_flag = False
                        #
                        # Tip: Ebike
                        for r in res[0][0]:
                            if r[4] > threshold:
                                if max_score_ebike < r[4]:
                                    max_score_ebike = r[4]
                                count_ebike += 1
                                break
                        if max_score_ebike > threshold:
                            ebike_flag = True
                        else:
                            ebike_flag = False
                        #
                        # Tip: Bike
                        for r in res[0][1]:
                            if r[4] > threshold:
                                if max_score_bike < r[4]:
                                    max_score_bike = r[4]
                                count_bike += 1
                                break
                        if max_score_bike > threshold:
                            bike_flag = True
                        else:
                            bike_flag = False
                        #
                        # Tip:
                        if count_ebike + count_bike > 1:
                            size_flag = True
                        else:
                            size_flag = False
                        #

                        #

                        if size_flag:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{}_{}_{}".format("multiError", timestamp, c,
                                                                               os.path.basename(image_name)))
                            cv2.imwrite(save_file_name, result_frame)
                            cv2.imshow("res:{}".format(args.model), result_frame)

                            #
                            cv2.imwrite(save_origin_file_name, raw_frame)

                        if ebike_flag:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{}_{}_{}".format("ebikeError", timestamp, c,
                                                                               os.path.basename(image_name)))
                            cv2.imwrite(save_file_name, result_frame)
                            cv2.imshow("res:{}".format(args.model), result_frame)
                            #
                            cv2.imwrite(save_origin_file_name, raw_frame)
                            # cv2.imshow("res_origin", raw_frame)


                        elif bike_flag:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{}_{}_{}".format("bikeError", timestamp, c,
                                                                               os.path.basename(image_name)))
                            cv2.imwrite(save_file_name, result_frame)
                            cv2.imshow("res:{}".format(args.model), result_frame)

                            #
                            cv2.imwrite(save_origin_file_name, raw_frame)

                        else:
                            save_file_name = os.path.join(save_folder,
                                                          "{}_{}_{}_{}".format("normal", timestamp, c,
                                                                               os.path.basename(image_name)))
                            cv2.imwrite(save_file_name, result_frame)
                            cv2.imshow("res:{}".format(args.model), result_frame)
                            #
                            cv2.imwrite(save_origin_file_name, raw_frame)
                            pass

                        ch = cv2.waitKey(1)
                        if ch == 27 or ch == ord('q') or ch == ord('Q'):
                            break
                    i += 1
                else:
                    i += 1
            else:
                break
    elif args.demo == 'big_videos_savexml':
        if os.path.isdir(args.path):
            files = get_video_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for file in files:
            print("".format(file))
            cap = cv2.VideoCapture(file)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            mkdir(local_rank, save_folder)
            mkdir(local_rank, os.path.join(save_folder, "origin"))

            i = 0
            threshold = 0.4
            #
            patch_w, patch_h = 5000, 5000
            patch_pos = [[0, 0]]

            while True:
                ret_val, raw_big_frame = cap.read()
                #
                if ret_val:
                    if i % 5 == 0:
                        for c, crop_pos in enumerate(patch_pos):
                            frame = raw_big_frame[crop_pos[1]:crop_pos[1] + patch_h,
                                    crop_pos[0]:crop_pos[0] + patch_w].copy()
                            #
                            raw_frame = None
                            raw_frame = frame.copy()
                            # cv2.imshow("raw{}".format(c), raw_frame)
                            #
                            meta, res = predictor.inference(frame)

                            #
                            image_name = "{}.jpg".format(i)
                            timestamp = int(time.time())
                            save_origin_file_name = os.path.join(save_folder, "origin",
                                                                 "{}_{}_{}".format(
                                                                     timestamp, c,
                                                                     os.path.basename(
                                                                         image_name)))
                            #
                            # print(save_origin_file_name)
                            result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False,
                                                               draw=True,
                                                               filename=save_origin_file_name.split('.jpg')[0],
                                                               xml=True)
                            # cv2.imshow("{}_crop".format(c), result_frame)
                            #
                            #
                            size_flag = False
                            #
                            count_ebike = 0
                            max_score_ebike = 0.
                            count_bike = 0
                            max_score_bike = 0.
                            #
                            ebike_flag = False
                            bike_flag = False
                            #
                            # Tip: Ebike
                            for r in res[0][0]:
                                if r[4] > threshold:
                                    if max_score_ebike < r[4]:
                                        max_score_ebike = r[4]
                                    count_ebike += 1
                                    break
                            if max_score_ebike > threshold:
                                ebike_flag = True
                            else:
                                ebike_flag = False
                            #
                            # Tip: Bike
                            for r in res[0][1]:
                                if r[4] > threshold:
                                    if max_score_bike < r[4]:
                                        max_score_bike = r[4]
                                    count_bike += 1
                                    break
                            if max_score_bike > threshold:
                                bike_flag = True
                            else:
                                bike_flag = False
                            #
                            # Tip:
                            if count_ebike + count_bike > 1:
                                size_flag = True
                            else:
                                size_flag = False
                            #

                            #

                            if size_flag:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("multiError", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)

                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)

                            if ebike_flag:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("ebikeError", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)
                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)
                                # cv2.imshow("res_origin", raw_frame)


                            elif bike_flag:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("bikeError", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)

                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)

                            else:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("normal", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)
                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)
                                pass

                            ch = cv2.waitKey(1)
                            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                                break
                        i += 1
                    else:
                        i += 1
                else:
                    break

    elif args.demo == 'big_videos_batch_savexml':
        if os.path.isdir(args.path):
            files = get_video_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for file in files:
            print("".format(file))
            cap = cv2.VideoCapture(file)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            mkdir(local_rank, save_folder)
            mkdir(local_rank, os.path.join(save_folder, "origin"))

            i = 0
            threshold = 0.4
            #
            patch_w, patch_h = 5000, 5000
            patch_pos = [[0, 0]]

            while True:
                ret_val, raw_big_frame = cap.read()
                #
                if ret_val:
                    if i % 5 == 0:
                        for c, crop_pos in enumerate(patch_pos):
                            frame = raw_big_frame[crop_pos[1]:crop_pos[1] + patch_h,
                                    crop_pos[0]:crop_pos[0] + patch_w].copy()
                            #
                            raw_frame = None
                            raw_frame = frame.copy()
                            # cv2.imshow("raw{}".format(c), raw_frame)
                            #
                            meta, res = predictor.inference(frame)

                            #
                            image_name = "{}.jpg".format(i)
                            timestamp = int(time.time())
                            save_origin_file_name = os.path.join(save_folder, "origin",
                                                                 "{}_{}_{}".format(
                                                                     timestamp, c,
                                                                     os.path.basename(
                                                                         image_name)))
                            #
                            # print(save_origin_file_name)
                            result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False,
                                                               draw=True,
                                                               filename=save_origin_file_name.split('.jpg')[0],
                                                               xml=True)
                            # cv2.imshow("{}_crop".format(c), result_frame)
                            #
                            #
                            size_flag = False
                            #
                            count_ebike = 0
                            max_score_ebike = 0.
                            count_bike = 0
                            max_score_bike = 0.
                            #
                            ebike_flag = False
                            bike_flag = False
                            #
                            # Tip: Ebike
                            for r in res[0][0]:
                                if r[4] > threshold:
                                    if max_score_ebike < r[4]:
                                        max_score_ebike = r[4]
                                    count_ebike += 1
                                    break
                            if max_score_ebike > threshold:
                                ebike_flag = True
                            else:
                                ebike_flag = False
                            #
                            # Tip: Bike
                            for r in res[0][1]:
                                if r[4] > threshold:
                                    if max_score_bike < r[4]:
                                        max_score_bike = r[4]
                                    count_bike += 1
                                    break
                            if max_score_bike > threshold:
                                bike_flag = True
                            else:
                                bike_flag = False
                            #
                            # Tip:
                            if count_ebike + count_bike > 1:
                                size_flag = True
                            else:
                                size_flag = False
                            #

                            #

                            if size_flag:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("multiError", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)

                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)

                            if ebike_flag:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("ebikeError", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)
                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)
                                # cv2.imshow("res_origin", raw_frame)


                            elif bike_flag:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("bikeError", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)

                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)

                            else:
                                save_file_name = os.path.join(save_folder,
                                                              "{}_{}_{}_{}".format("normal", timestamp, c,
                                                                                   os.path.basename(image_name)))
                                cv2.imwrite(save_file_name, result_frame)
                                # cv2.imshow("res:{}".format(args.model), result_frame)
                                #
                                cv2.imwrite(save_origin_file_name, raw_frame)
                                pass

                            ch = cv2.waitKey(1)
                            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                                break
                        i += 1
                    else:
                        i += 1
                else:
                    break

    elif args.demo == 'big_image_batch_savexml':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()

        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        mkdir(local_rank, os.path.join(save_folder, "origin"))

        i = 0
        for file in files:
            i += 1
            print("{}".format(file))
            cap = cv2.imread(file)

            threshold = 0.4
            height, width, channel = cap.shape

            patch_w, patch_h = 512, 512
            strides = 400

            patch_pos = [[0, 0]]
            y1, y2 = 0, patch_h
            w_num_image = int((width - 112) / strides) + 1
            h_num_image = int((height - 112) / strides) + 1

            tmp = 0
            tmp1 = 0
            for current_epoch in range(w_num_image * h_num_image):
                x1 = 0 + tmp1 * strides
                if x1 + strides > width:
                    x2 = width
                    x1 = x2 - patch_w
                else:
                    x2 = x1 + patch_w
                tmp1 += 1
                if x2 == width:
                    tmp1 = 0
                    tmp += 1
                    y1 = 0 + tmp * strides
                    if y1 + strides > height:
                        y2 = height
                        y1 = y2 - patch_h
                    else:
                        y2 = y1 + patch_h

                frame = cap[y1:y2, x1:x2].copy()
                # 推理
                raw_frame = None
                raw_frame = frame.copy()
                # cv2.imshow("raw{}".format(c), raw_frame)
                #
                meta, res = predictor.inference(frame)

                #
                image_name = "{}.jpg".format(i)
                timestamp = int(time.time())
                save_origin_file_name = os.path.join(save_folder, "origin",
                                                     "{}_{}_{}".format(
                                                         timestamp, current_epoch,
                                                         os.path.basename(
                                                             image_name)))
                #
                # print(save_origin_file_name)
                result_frame = predictor.visualize(res[0], meta, cfg.class_names, threshold, show=False,
                                                    filename=save_origin_file_name.split('.jpg')[0],
                                                   xml=True)
                size_flag = False
                #
                count_ebike = 0
                max_score_ebike = 0.
                count_bike = 0
                max_score_bike = 0.
                #
                ebike_flag = False
                bike_flag = False
                #
                # Tip: Ebike
                for r in res[0][0]:
                    if r[4] > threshold:
                        if max_score_ebike < r[4]:
                            max_score_ebike = r[4]
                        count_ebike += 1
                        break
                if max_score_ebike > threshold:
                    ebike_flag = True
                else:
                    ebike_flag = False
                #
                # Tip: Bike
                for r in res[0][1]:
                    if r[4] > threshold:
                        if max_score_bike < r[4]:
                            max_score_bike = r[4]
                        count_bike += 1
                        break
                if max_score_bike > threshold:
                    bike_flag = True
                else:
                    bike_flag = False
                #
                # Tip:
                if count_ebike + count_bike > 1:
                    size_flag = True
                else:
                    size_flag = False
                #

                #

                if size_flag:
                    save_file_name = os.path.join(save_folder,
                                                  "{}_{}_{}_{}".format("multiError", timestamp, current_epoch,
                                                                       os.path.basename(image_name)))
                    cv2.imwrite(save_file_name, result_frame)
                    cv2.imshow("res:{}".format(args.model), result_frame)

                    #
                    cv2.imwrite(save_origin_file_name, raw_frame)

                if ebike_flag:
                    save_file_name = os.path.join(save_folder,
                                                  "{}_{}_{}_{}".format("ebikeError", timestamp, current_epoch,
                                                                       os.path.basename(image_name)))
                    cv2.imwrite(save_file_name, result_frame)
                    cv2.imshow("res:{}".format(args.model), result_frame)
                    #
                    cv2.imwrite(save_origin_file_name, raw_frame)
                    # cv2.imshow("res_origin", raw_frame)


                elif bike_flag:
                    save_file_name = os.path.join(save_folder,
                                                  "{}_{}_{}_{}".format("bikeError", timestamp, current_epoch,
                                                                       os.path.basename(image_name)))
                    cv2.imwrite(save_file_name, result_frame)
                    cv2.imshow("res:{}".format(args.model), result_frame)

                    #
                    cv2.imwrite(save_origin_file_name, raw_frame)

                else:
                    save_file_name = os.path.join(save_folder,
                                                  "{}_{}_{}_{}".format("normal", timestamp, current_epoch,
                                                                       os.path.basename(image_name)))
                    cv2.imwrite(save_file_name, result_frame)
                    cv2.imshow("res:{}".format(args.model), result_frame)
                    #
                    cv2.imwrite(save_origin_file_name, raw_frame)
                    pass


if __name__ == '__main__':
    main()
