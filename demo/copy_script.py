import os
import shutil


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_path_list(stack, path):
    ima_list = []
    if stack == 0:
        for root, dirs, files in os.walk(path):
            for i in files:
                if i.split(sep=".")[-1] in ["jpg"]:
                    path = os.path.join(root, i)
                    ima_list.append(path)
    else:
        for root, dirs, files in os.walk(path):
            for i in files:
                i = i.replace("normal_", '')
                if i.split(sep=".")[-1] in ["jpg"]:
                    # path = os.path.join(root, i)
                    ima_list.append(i)

    print("file length:", len(ima_list))

    return ima_list


def get_require_list(img_list, normal_list):
    res_list = []
    for i in img_list:
        if i.split('/')[-1] not in normal_list:
            # print(i)
            res_list.append(i)
    print("file length: ", len(res_list))
    return res_list


def get_error_list(path):
    l = os.listdir(path)
    res = []
    for i in l:
        if "error" in i:
            res += os.listdir(os.path.join(path, i))
    print("file length:", len(res))
    return res


def run():
    path = '/data1/wl/detect/2021_07_05_18_59_54'
    img_path = '/data1/wl/SUN397'
    save_path = "/data1/wl/detect"
    normal_list = get_path_list(1, path + "/normal")
    img_list = get_path_list(0, img_path)

    img_list = get_require_list(img_list, normal_list)  # with path
    require_list = get_error_list(path)
    mkdir(os.path.join(save_path, "multi_error"))
    mkdir(os.path.join(save_path, "ebike_error"))
    mkdir(os.path.join(save_path, "bike_error"))
    for i in img_list:
        print(i)
        name = i.split("/")[-1]
        if "multi_error_" + name in require_list:
            shutil.copyfile(i, os.path.join(save_path, "multi_error", name))
        elif "ebike_error_" + name in require_list:
            shutil.copyfile(i, os.path.join(save_path, "ebike_error", name))
        elif "bike_error_" + name in require_list:
            shutil.copyfile(i, os.path.join(save_path, "bike_error", name))
        else:
            pass


run()
