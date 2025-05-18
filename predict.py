import time

import cv2
import numpy as np
from PIL import Image
import os
import argparse

from segformer import SegFormer_Segmentation
from utils.value import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'single_predict'    表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'batch_predict'     表示批量图片预测，输出分割指标dice、hd95，检测指标mIou、mAP@0.5、mAP@0.5:0.95
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #----------------------------------------------------------------------------------------------------------#
    parser.add_argument('--mode', type=str, default='batch_predict', help='test mode')
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    parser.add_argument('--count', type=bool, default=False, help='count pixel')
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = 'xxx.mp4'即可，代表读取出根目录下的xxx.mp4文件。
    #----------------------------------------------------------------------------------------------------------#
    parser.add_argument('--video_path', type=str, default='0', help='video_path')
    #----------------------------------------------------------------------------------------------------------#
    #   video_save_path     表示视频保存的路径，当video_save_path=''时表示不保存
    #                       想要保存视频，则设置如video_save_path = 'yyy.mp4'即可，代表保存为根目录下的yyy.mp4文件。
    #----------------------------------------------------------------------------------------------------------#
    parser.add_argument('--video_save_path', type=str, default='', help='video_save_path')
    #----------------------------------------------------------------------------------------------------------#
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    parser.add_argument('--video_fps', type=int, default=25, help='video_fps')
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #----------------------------------------------------------------------------------------------------------#
    parser.add_argument('--test_interval', type=int, default=100, help='test_interval')
    #----------------------------------------------------------------------------------------------------------#
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    parser.add_argument('--fps_image_path', type=str, default='img/street.jpg', help='fps_image_path')
    #-------------------------------------------------------------------------#
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    parser.add_argument('--dir_save_path', type=str, default='img_out/', help='dir_save_path')
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #-------------------------------------------------------------------------#
    parser.add_argument('--simplify', type=bool, default=True, help='simplify onnx')
    #-------------------------------------------------------------------------#
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    parser.add_argument('--onnx_save_path', type=str, default='model_data/segformer.onnx', help='onnx_save_path')
    #-------------------------------------------------------------------------#
    #   img_dir      储存测试照片文件的地址
    #-------------------------------------------------------------------------#
    parser.add_argument('--img_dir', type=str, default='X:\segformer-pytorch-master\datasets\CrackTree\JPEGImages', help='img_dir') 
    #-------------------------------------------------------------------------#
    #   label_dir      储存掩码标签的地址
    #-------------------------------------------------------------------------#
    parser.add_argument('--label_dir', type=str, default='X:\segformer-pytorch-master\datasets\CrackTree\SegmentationClass', help='label_dir')
    #-------------------------------------------------------------------------#
    #   test_output    输出测试结果的文件地址
    #-------------------------------------------------------------------------#
    parser.add_argument('--test_output', type=str, default=r'X:\segformer-pytorch-master\pred_result\cracktree', help='test_output')
    #-------------------------------------------------------------------------#
    #   list_path      储存文件索引内容的txt地址
    #-------------------------------------------------------------------------#
    parser.add_argument('--list_path', type=str, default=r'X:\segformer-pytorch-master\datasets\CrackTree\list\val.txt', help='list_path')
    #-------------------------------------------------------------------------#
    #   model_path指向model_data文件夹下的权值文件
    #-------------------------------------------------------------------------#
    parser.add_argument('--model_path', type=str, default=r'X:\segformer-pytorch-master\model_data\cracktree\b2\1\best_epoch_weights.pth', help='model_path')
    #-------------------------------------------------------------------------#
    #   所需要区分的类的个数+1
    #-------------------------------------------------------------------------#
    parser.add_argument('--num_classes', type=int, default=2, help='num_classes')
    #-------------------------------------------------------------------------#
    #   所使用的的主干网络：
    #   b0、b1、b2、b3、b4、b5
    #-------------------------------------------------------------------------#
    parser.add_argument('--phi', type=str, default='b2', help='phi')
    #-------------------------------------------------------------------------#
    #   mix_type参数用于控制检测结果的可视化方式
    #   mix_type = 0的时候代表原图与生成的图进行混合
    #   mix_type = 1的时候代表仅保留生成的图
    #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
    #-------------------------------------------------------------------------#
    parser.add_argument('--mix_type', type=int, default=0, help='mix_type')

    args = parser.parse_args()
    mode = args.mode
    count = args.count
    video_path = args.video_path
    video_save_path = args.video_save_path
    video_fps = args.video_fps
    test_interval = args.test_interval
    fps_image_path = args.fps_image_path
    dir_save_path = args.dir_save_path
    simplify = args.simplify
    onnx_save_path = args.onnx_save_path
    img_dir = args.img_dir
    label_dir = args.label_dir
    test_output = args.test_output
    list_path = args.list_path
    model_path = args.model_path
    num_classes = args.num_classes
    phi = args.phi
    mix_type = args.mix_type

    # name_classes = ['background', 'crack', 'pothole', 'patch']
    name_classes = ['background', 'crack']
    segformer = SegFormer_Segmentation(model_path=model_path, num_classes=num_classes, phi=phi, mix_type=mix_type)

    if mode == 'single_predict':
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = segformer.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == 'batch_predict':
        with open(list_path, 'r', encoding='utf-8') as file:
            idx = 1
            metric_list = []
            for line in file.readlines():
                line = line.strip()
                image_path = os.path.join(img_dir, f'{line}.jpg')
                label_path = os.path.join(label_dir, f'{line}.png')
                image = Image.open(image_path)
                label = Image.open(label_path)
                prediction = segformer.get_miou_png(image)
                metric_i = []
                for i in range(1, num_classes):
                    # prediction = erode_mask(prediction)
                    result = calculate_metric_percase(prediction, np.array(label), i)
                    if result is not None:
                        metric_i.append(result)
                print(f'idx {idx} case {line} mean_dice {np.mean(metric_i, axis=0)[0]} mean_hd95 {np.mean(metric_i, axis=0)[1]} iou {np.mean(metric_i, axis=0)[2]} mAP@{np.mean(metric_i, axis=0)[3]} {np.mean(metric_i, axis=0)[4]} mAP@0.5:0.95 {np.mean(metric_i, axis=0)[5]}')
                idx += 1
                for item in metric_i:
                    metric_list.append(item)

                image = np.array(image)
                label = np.array(label)
                sample = {'image':image, 'pred':prediction, 'label':label}
                output_file = os.path.join(test_output, f'{line}.npz')
                np.savez(output_file, **sample)

        metric_arr = np.array(metric_list)
        for i in range(1, num_classes):
            rows_with_label = metric_arr[metric_arr[:, -1] == i]
            mean_result = np.mean(rows_with_label, axis=0)
            print(f'Mean class {i} mean_dice {mean_result[0]} mean_hd95{mean_result[1]} iou {mean_result[2]} map@{mean_result[3]} {mean_result[4]} map@0.5:0.95 {mean_result[5]}')
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        iou = np.mean(metric_list, axis=0)[2]
        mAP = np.mean(metric_list, axis=0)[4]
        print(f'Testing performance in best val model: mean_dice : {performance} mean_hd95 : {mean_hd95} iou : {iou} mAP@{np.mean(metric_i, axis=0)[3]} {mAP} mAP@0.5:0.95 {np.mean(metric_list, axis=0)[5]}')
        print('Testing Finished!')

    elif mode == 'video':
        capture=cv2.VideoCapture(video_path)
        if video_save_path!='':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError('未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。')

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(segformer.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print('fps= %.2f'%(fps))
            frame = cv2.putText(frame, 'fps= %.2f'%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('video',frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!='':
                out.write(frame)

            if c==27:
                capture.release()
                break
        print('Video Detection Done!')
        capture.release()
        if video_save_path!='':
            print('Save processed video to the path :' + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == 'fps':
        img = Image.open(fps_image_path)
        tact_time = segformer.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == 'export_onnx':
        segformer.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError('Please specify the correct mode: single_predict, batch_predict, video or fps')
