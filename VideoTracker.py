import cv2 as cv
import os
import time
import numpy as np
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, VGG16
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt



class boundingBox:     #边界框  
    def __init__(self, xc, yc, w, h):   # x左上角，y左上角，宽度，高度
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h


class parameter:
    def __init__(self, resize, feature_type, sigma, search_area_scale_factor, scale_factors, eta):
        self.resize = resize
        self.feature_type = feature_type
        self.sigma = sigma
        self.search_area_scale_factor = search_area_scale_factor
        self.scale_factors = scale_factors
        self.eta = eta

        if self.feature_type == 'deep':   # 创建模型以提取深度特征
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:   # 限制 TensorFlow 仅在第一个 GPU 上分配 1*X GB 内存
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 4))])
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    print(e)   # 必须在 GPU 初始化之前设置虚拟设备
            # InceptionV3
            self.model = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.resize[0], self.resize[0], 3))
            self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[5].output)
            # VGG16
            # self.model = VGG16(weights='imagenet', include_top=False, input_shape=(self.params.resize[0], self.params.resize[0], 3))
            # self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-5].output)
        else:
            self.model = None


class trackFrame:
    def __init__(self, params):
        '''params = parameter(  resize=(224,224), 
                                        feature_type='intensity', 
                                        sigma=3.0, 
                                        search_area_scale_factor=2.0, 
                                        scale_factors=[0.98, 1.00, 1.02], 
                                        eta=0.15)
        '''
        self.params = params    

        # 获取余弦窗口（用于标准化）
        self.window = self.cosine_window(self.params.resize[0], self.params.resize[1])

        # 生成高斯响应函数及其傅里叶 
        self.gi = self.gaussian_response_func(self.params.resize[0], self.params.resize[1], self.params.sigma)
        self.Gi = np.fft.fft2(self.gi)  #FFT: Fast Fourier Transform

    def initialize(self, frame, bounding_box):
        self.bounding_box = bounding_box
        _, fourier, fourier_conjugate =  self.extracted_features(frame, self.bounding_box, self.window, 1.0, self.params)  # 获取提取的特征
        if fourier.ndim > 2:
            self.Gi = np.repeat(np.expand_dims(self.Gi, axis=2), fourier.shape[2], axis=2)   # 重塑高斯响应函数的傅里叶（如果需要）
        self.Ai = self.Gi * fourier_conjugate   # 初始化过滤器部分
        self.Bi = fourier * fourier_conjugate

    def gaussian_response_func(self, height, width, sigma):
        y = np.expand_dims(np.arange(height), axis=1)
        x = np.expand_dims(np.arange(width), axis=0)
        gaussian_resp_row = np.exp(-np.square(y - height / 2) / (2 * np.square(sigma)))
        gaussian_resp_col = np.exp(-np.square(x - width / 2) / (2 * np.square(sigma)))
        gaussian_resp = gaussian_resp_row @ gaussian_resp_col
        return gaussian_resp

    def cosine_window(self, height, width):
        window_row = np.expand_dims(np.hanning(height), axis=1)
        window_col = np.expand_dims(np.hanning(width), axis=0)
        window = window_row @ window_col
        return window

    def preprocess(self, sample, window):
        height, width = sample.shape
        sample = np.log(sample + 1)
        sample = (sample - np.mean(sample)) / (np.std(sample) + 1e-5)  # Normalise image
        sample = sample * window  # Apply windowing on image
        return sample

    def crop(self, frame, bounding_box, curr_scale, params):
        w = bounding_box.w * curr_scale * params.search_area_scale_factor
        h = bounding_box.h * curr_scale * params.search_area_scale_factor
        x_tl = int(bounding_box.xc - w / 2)   # 计算样本边界
        x_br = int(x_tl + w)
        y_tl = int(bounding_box.yc - h / 2)
        y_br = int(y_tl + h)
        if x_tl < 0:           # 计算所需的填充
            x_l_pad = -x_tl
        else:
            x_l_pad = 0

        if x_br > frame.shape[1] - 1:
            x_r_pad = x_br - frame.shape[1]
        else:
            x_r_pad = 0

        if y_tl < 0:
            y_t_pad = - y_tl
        else:
            y_t_pad = 0

        if y_br > frame.shape[0] - 1:
            y_b_pad = y_br - frame.shape[0]
        else:
            y_b_pad = 0
        frame_padded = np.pad(frame, [(y_t_pad, y_b_pad), (x_l_pad, x_r_pad)], 'edge')   # 为帧添加所需的填充
        sample = frame_padded[y_tl+y_t_pad:y_br+y_t_pad,x_tl+x_l_pad:x_br+x_l_pad]       # 裁剪样本
        sample_resized = cv.resize(sample, params.resize, interpolation=cv.INTER_AREA)   # 将样本大小调整为所需大小
        return sample_resized

    def extracted_features(self, frame, bounding_box, window, curr_scale, params):
        features = None

        # 获取裁剪样本
        sample = self.crop(frame, bounding_box, curr_scale, params)

        # 预处理样本
        if params.feature_type == 'intensity':
            features = self.preprocess(sample, window)
        elif params.feature_type == 'gradient':
            sample = self.preprocess(sample, window)
            features = np.gradient(np.gradient(sample, axis=0), axis=1)
        elif params.feature_type == 'hog':
            sample = self.preprocess(sample, window)
            _, features = hog(sample, orientations=9, pixels_per_cell=(8,8),
                                    cells_per_block=(1,1), visualize=True, multichannel=False, block_norm='L2')
        elif params.feature_type == 'deep':
            # Repeat intensity channel to RGB  将强度通道重复到 RGB
            sample = np.expand_dims(sample, axis=(0,3))
            sample = np.repeat(sample, 3, axis=3)
            sample = preprocess_input(sample)  # 预处理样本
            features = np.squeeze(params.model.predict(sample))  # 提取特征
            features = cv.resize(features, params.resize, interpolation=cv.INTER_AREA)  # 将特征调整为所需大小
            features = (features - np.mean(features)) / (np.std(features) + 1e-5)  # 归一化特征
            features = features * np.repeat(np.expand_dims(window, axis=2), features.shape[2], axis=2)  # 对特征实现窗口化

        # 计算特征傅里叶
        features_fourier = np.fft.fft2(features,axes=(0,1))
        features_fourier_conjugate = np.conjugate(features_fourier)
        return features, features_fourier, features_fourier_conjugate

    def track(self, frame):
        best_peak_score = 0
        best_scale = -1
        y_disp = 0
        x_disp = 0
        tracking_failure = False

        # PSR (Peak-to-Sidelobe Ratio) 阈值用于检测跟踪故障
        psr = 2

        for iScale, scale in enumerate(self.params.scale_factors):   # [0.98,1.00,1.02]
            _, fourier, fourier_conjugate =  self.extracted_features(frame, self.bounding_box, self.window, scale, self.params)  #特征提取

           # 获取响应图及其空间形式
            Gr = (self.Ai / self.Bi) * fourier    #Hi=Ai/Bi, G=H·F
            if fourier.ndim > 2:
                Gr = np.sum(Gr, axis=2)
            gr = np.real(np.fft.ifft2(Gr))

            # 找到峰值位置
            peak_score = np.max(gr)
            peak_position = np.where(gr == peak_score)

            if peak_score > best_peak_score:
                best_peak_score = peak_score
                best_scale = scale
                y_disp = int(np.mean(peak_position[0]) - gr.shape[0] / 2)
                x_disp = int(np.mean(peak_position[1]) - gr.shape[1] / 2)

                # 查找给定比例的 PSR（具有 21x21 旁瓣，5x5 中心区域）
                gr_padded = np.pad(gr, [(10, 10), (10, 10)], 'edge')
                peak_position_padded = (int(peak_position[0] + 10), int(peak_position[1] + 10))
                sidelobe_vector = np.concatenate((gr_padded[peak_position_padded[0]-10:peak_position_padded[0]-5, peak_position_padded[1]-10:peak_position_padded[1]+10].flatten(),
                                                gr_padded[peak_position_padded[0]+5:peak_position_padded[0]+10, peak_position_padded[1]-10:peak_position_padded[1]+10].flatten(),
                                                gr_padded[peak_position_padded[0]-10:peak_position_padded[0]+10, peak_position_padded[1]-10:peak_position_padded[1]-5].flatten(),
                                                gr_padded[peak_position_padded[0]-10:peak_position_padded[0]+10, peak_position_padded[1]+5:peak_position_padded[1]+10].flatten()))
                sidelobe_mean = np.mean(sidelobe_vector)
                sidelobe_std = np.std(sidelobe_vector)

                if (peak_score - sidelobe_mean) / sidelobe_std < psr:
                    tracking_failure = True
                else:
                    tracking_failure = False

        # 计算尺寸调整的乘数
        y_scale_multiplier = self.bounding_box.h * best_scale * self.params.search_area_scale_factor / self.params.resize[0]
        x_scale_multiplier = self.bounding_box.w * best_scale * self.params.search_area_scale_factor / self.params.resize[1]

        # 更新目标位置
        self.bounding_box.yc = self.bounding_box.yc + y_disp * y_scale_multiplier
        self.bounding_box.xc = self.bounding_box.xc + x_disp * x_scale_multiplier

        # 更新目标尺寸
        self.bounding_box.h = best_scale * self.bounding_box.h
        self.bounding_box.w = best_scale * self.bounding_box.w

        if not tracking_failure:   # 获取提取的特征（再次找到本地化）
            _, fourier, fourier_conjugate =  self.extracted_features(frame, self.bounding_box, self.window, 1.0, self.params)   
            self.Ai = self.params.eta * (self.Gi * fourier_conjugate) + (1 - self.params.eta) * self.Ai   # 更新过滤器
            self.Bi = self.params.eta * (fourier * fourier_conjugate) + (1 - self.params.eta) * self.Bi
        
        return self.bounding_box


class readFrame:
    def __init__(self, data_path):
        self.data_path = data_path
        self.index = 0
        self.resolution = None 
        if os.path.isdir(self.data_path):   # 输入方式：图像序列
            _, _, self.image_files = next(os.walk(self.data_path))
            self.size = len(self.image_files)
            self.read_function = self.read_image_sequence_frame
        else:   # 输入方式：视频
            self.video_capture = cv.VideoCapture(self.data_path)
            self.size = self.video_capture.get(cv.CAP_PROP_FRAME_COUNT)
            self.read_function = self.read_video_frame
            height = self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
            width = self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
            self.resolution = str(int(width)) + "×" + str(int(height))

    def read_image_sequence_frame(self):
        current_frame = None
        if self.index < self.size:
            current_frame = cv.imread(os.path.join(self.data_path, self.image_files[self.index]))  # 读取当前帧
            self.index = self.index + 1
        return current_frame

    def read_video_frame(self):
        current_frame = None
        if self.index < self.size and self.video_capture.isOpened():
            ret, current_frame = self.video_capture.read()  # 读取当前帧
            self.index = self.index + 1
        else:
            self.video_capture.release()
        return current_frame

    def read_frame(self):
        return self.read_function()

    def good(self):
        return self.index < self.size

class mainTrack:

    def __init__(self,data_path,featureType):
        self.fps = []
        self.data_path = data_path
        ground_truth_path = None
        showFigure = True
        # 获取跟踪器参数
        self.params = parameter( resize=(224,224), 
                            feature_type=featureType, 
                            sigma=3.0, 
                            search_area_scale_factor=2.0, 
                            scale_factors=[0.98, 1.00, 1.02], 
                            eta=0.15)
        target_bounding_boxes = self.startTrack(params=self.params, save=True, ground_truth_path=ground_truth_path)
        if ground_truth_path is not None:  # 衡量跟踪器性能
            self.trackerPerformance(target_bounding_boxes, ground_truth_path)
        if showFigure == True:
            self.drawFigure()

    def initializeTracker(self, current_frame, current_frame_grey, params, ground_truth_path):
        if ground_truth_path is None:
            ''' ROI (Region of Interest)
                感兴趣区域 (ROI) 是用户想要过滤或操作的图像部分
            '''
            bb = np.array(cv.selectROI('trackFrame', current_frame, False, False), dtype=int)   # 提示选择 ROI
            target_bounding_box = boundingBox(bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3])   #xc,yc,w,h
        else:
            true_target_bounding_box = []
            with open(ground_truth_path, 'r') as f:
                for line in f:
                    numbers = line.split()
                    numbers = numbers[1:5]
                    for number in numbers:
                        true_target_bounding_box.append(int(number))
                    break
            target_bounding_box = boundingBox  ((true_target_bounding_box[0] + true_target_bounding_box[2]) / 2, 
                                                (true_target_bounding_box[1] + true_target_bounding_box[3]) / 2,
                                                true_target_bounding_box[2] - true_target_bounding_box[0], 
                                                true_target_bounding_box[3] - true_target_bounding_box[1]
                                            )
        tracker = trackFrame(params)  #初始化追踪器
        tracker.initialize(current_frame_grey, target_bounding_box)
        return params, tracker, target_bounding_box


    def startTrack(self, params, save, ground_truth_path):
        data_path = self.data_path
        frame_reader = readFrame(data_path)   # 初始化帧阅读器
        frame_counter = 0
        self.frame_reader = frame_reader

        tracker = None
        target_bounding_box = None
        video_writer = None
        target_bounding_boxes = []
        total_time = 0.0

        while(frame_reader.good()):
            current_frame = frame_reader.read_frame()  # 读取当前帧
            current_frame_grey = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY).astype(np.float32)  # 转换为灰度

            if frame_counter == 0:   # 如果是第一帧，则初始化跟踪器
                params, tracker, target_bounding_box = self.initializeTracker(current_frame, current_frame_grey, params, ground_truth_path)
                if save:  # 初始化视频写入器（如果启用）
                    video_writer = cv.VideoWriter('result/'+data_path.replace('.mp4','').replace('video/','')+'_output.mp4', cv.VideoWriter_fourcc('M','J','P','G'), 30, (current_frame.shape[1], current_frame.shape[0]))
                fps = 0.0
            else:
                start_time = time.time()
                target_bounding_box = tracker.track(current_frame_grey)  # 获取当前目标定位
                finish_time = time.time()
                total_time = total_time + (finish_time - start_time)  # 计算 FPS
                fps = 1 / (finish_time - start_time)
                self.fps.append(fps)

            frame_counter = frame_counter + 1   # 增加帧计数器
            curr_frame_disp = np.copy(current_frame)   # 可视化当前帧和目标定位
            xtl = int(target_bounding_box.xc - target_bounding_box.w / 2)  #xmin
            ytl = int(target_bounding_box.yc - target_bounding_box.h / 2)  #ymin
            xbr = int(xtl + target_bounding_box.w)                         #xmax
            ybr = int(ytl + target_bounding_box.h)                         #ymax
            cv.rectangle(curr_frame_disp, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)
            cv.putText( curr_frame_disp, 
                        'FPS={:0.2f}'.format(fps), 
                        org=(15, 15), 
                        fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, 
                        color=(0, 0, 255), 
                        thickness=1)              # 显示FPS
            cv.imshow('trackFrame', curr_frame_disp)
            key = cv.waitKey(33)

            if save:  # 保存当前帧（如果启用）
                video_writer.write(curr_frame_disp)
            if key == ord('q'):    # 键盘功能：Exit
                break   
            elif key == ord('r'):  # 键盘功能：重新初始化跟踪器
                params, tracker, target_bounding_box = self.initializeTracker(current_frame, current_frame_grey, params, ground_truth_path)

            xtl = int(target_bounding_box.xc - target_bounding_box.w / 2)
            ytl = int(target_bounding_box.yc - target_bounding_box.h / 2)
            target_bounding_boxes.append([xtl, ytl, target_bounding_box.w, target_bounding_box.h])
        
        if video_writer is not None:  # 关闭视频编写器
            video_writer.release()
        cv.destroyAllWindows()  # 关闭窗口

        print('------'+data_path+'------')
        print('Mean FPS = ' + '{:0.2f}'.format(frame_counter / total_time))
        print('Frame Count =', frame_reader.size)
        if not frame_reader.resolution == None:
            print('Video Resolution =', frame_reader.resolution)
        return target_bounding_boxes


    def trackerPerformance(self, predTargetBoundingBoxes, groundTruthPath):
        trueTargetBoundingBoxes = []

        with open(groundTruthPath, 'r') as f:
            for line in f:
                trueTargetBoundingBox = []

                numbers = line.split()
                numbers = numbers[1:5]

                for number in numbers:
                    trueTargetBoundingBox.append(int(number))

                temp = trueTargetBoundingBox
                trueTargetBoundingBox = [(temp[0] + temp[2])/2, (temp[1] + temp[3])/2,
                                        temp[2] - temp[0], temp[3] - temp[1]]

                trueTargetBoundingBoxes.append(trueTargetBoundingBox)

        avgAcc_iou = 0   # 指标 1：使用intersection-over-union找到的平均准确率
        eao = 0          # 指标 2：预期平均重叠
        number_of_eao_curves = 10
        eao_vector = np.zeros(number_of_eao_curves)

        noOfFrames = len(predTargetBoundingBoxes)

        for iFrame in range(noOfFrames):
            predTargetBoundingBox = predTargetBoundingBoxes[iFrame]
            trueTargetBoundingBox = trueTargetBoundingBoxes[iFrame]

            dx = min(predTargetBoundingBox[0] + predTargetBoundingBox[2], trueTargetBoundingBox[0] + trueTargetBoundingBox[2]) - \
                max(predTargetBoundingBox[0], trueTargetBoundingBox[0])

            dy = min(predTargetBoundingBox[1] + predTargetBoundingBox[3], trueTargetBoundingBox[1] + trueTargetBoundingBox[3]) - \
                max(predTargetBoundingBox[1], trueTargetBoundingBox[1])

            if dx > 0 and dy > 0:
                intersection = dx * dy
            else:
                intersection = 0

            union = predTargetBoundingBox[2] * predTargetBoundingBox[3] + trueTargetBoundingBox[2] * trueTargetBoundingBox[3] - intersection
            acc_forIthFrame_iou = intersection / union
            avgAcc_iou += acc_forIthFrame_iou / noOfFrames

            if iFrame >= (noOfFrames - number_of_eao_curves):
                eao_vector[iFrame - (noOfFrames - number_of_eao_curves)] = avgAcc_iou

        eao = np.sum(eao_vector) / number_of_eao_curves

        print('Accuracy w.r.t intersection-over-union = ' + '{:0.3f}'.format(avgAcc_iou))
        print('EAO = ' + '{:0.3f}'.format(eao))

    def drawFigure(self):
        x = list(range(1, int(self.frame_reader.size)))
        y = self.fps
        plt.plot(x,y)
        plt.xlabel('Frame count')
        plt.ylabel('Frames per second (FPS)')
        plt.title(self.data_path)
        plt.savefig('figure/'+data_path.replace('.mp4','').replace('video/','')+'_'+self.params.feature_type+'_fps.jpg')
        plt.show()
        

if __name__ == "__main__" :
    data_path = 'video/dance.mp4'
    feature_type = 'intensity'   #可自行更改：intensity / hog / gradient
    main = mainTrack(data_path, feature_type)  #开始追踪
    

