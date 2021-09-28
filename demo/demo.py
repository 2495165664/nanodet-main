import cv2
import os
import time
import torch
import argparse
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.util.path import mkdir

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
videos_ext = ['.mp4', '.mov', '.avi', '.mkv']


def get_video_list(path):
    video_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in videos_ext:
                video_names.append(apath)
    return video_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='image', help='demo type, eg. image, video and webcam')
    parser.add_argument('--config', default="../config/person.yml", help='model config file path')
    parser.add_argument('--model', default="", help='model file path')
    parser.add_argument('--path', default='/data1/wl/train_data/p/val/imgs', help='path to images or video')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    parser.add_argument('--save_result', default="/data1/wl/detect/images/preson", action='store_false', help='whether to save the inference result of image/video')
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

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=False)
        print('viz time: {:.3f}s'.format(time.time()-time1))
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
        ecount = 0
        ncount = 0
        threshold = 0.7
        for image_name in files:
            # print()
            meta, res = predictor.inference(image_name)
            print(res)
            for i in res[0][0]:
                # print(i)
                if i[4] >= threshold:
                    ecount += 1
                    break
            # for i in res[0][1]:
            #     if i[4] >= threshold:
            #         ncount += 1
            #         break

            result_image = predictor.visualize(res[0], meta, cfg.class_names, threshold)

            #
            size_flag = False
            count_ebike = 0
            count_bike = 0
            #
            ebike_flag = False
            bike_flag = False
            # Tip:
            # Tip:
            for r in res[0][0]:
                if r[4] > threshold:
                    ebike_flag = True
                    count_ebike += 1
                    break
                else:
                    ebike_flag = False

            # Tip:
            # for r in res[0][1]:
            #     if r[4] > threshold:
            #         bike_flag = True
            #         count_bike += 1
            #         break
            #     else:
            #         bike_flag = False
            # Tip:
            if count_ebike + count_bike > 1:
                size_flag = True
            else:
                size_flag = False
            # print(res)

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
        print(len(files))
        print("ebike count: ", ecount)
        print("neg count: ", ncount)

    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        mkdir(local_rank, save_folder)
        save_path = os.path.join(save_folder, args.path.split('/')[-1]) if args.demo == 'video' else os.path.join(save_folder, 'camera.mp4')
        print(f'save_path is {save_path}')
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = predictor.inference(frame)
                result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                if args.save_result:
                    vid_writer.write(result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break
            else:
                break
    elif args.demo == 'videos':
        files = get_video_list(args.path)
        for file in files:
            cap = cv2.VideoCapture(file)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            save_folder = os.path.join(args.save_result, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            mkdir(local_rank, save_folder)
            save_path = os.path.join(save_folder, file.split("/")[-1])
            print(f'save_path is {save_path}')
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
            while True:
                ret_val, frame = cap.read()
                if ret_val:
                    meta, res = predictor.inference(frame)
                    result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                    if args.save_result:
                        vid_writer.write(result_frame)
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord('q') or ch == ord('Q'):
                        break
                else:
                    break


if __name__ == '__main__':
    main()
