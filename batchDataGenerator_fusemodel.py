import numpy as np
from keras.utils import Sequence
import os
import cv2
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from scipy.ndimage import gaussian_filter1d
import random


class BatchPPGGenerator(Sequence):
    def __init__(self, npz_dir,
                 list_files,
                 img_dim=(128, 128, 3),
                 PPG_dim=(60,),
                 batch_size=32,
                 shuffle=True):
        self.npz_dir = npz_dir
        self.list_files = list_files
        self.img_dim = img_dim
        self.PPG_dim = PPG_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_files) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Find list of IDs
        list_files_temp = [self.list_files[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_files_temp)
        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def PPG_normalization(self, PPG):
        min_PPG = np.min(PPG)
        max_PPG = np.max(PPG)
        PPG_norm = (PPG - min_PPG) / (max_PPG - min_PPG)
        return PPG_norm
    
    def PPG_normalization_V2(self, PPG):
        """
        :param PPG:
        :return: PPG_norm --> (0, 1)
        """
        min_PPG = np.min(PPG)
        max_PPG = np.max(PPG)
        PPG_norm = (PPG - min_PPG) / (max_PPG - min_PPG)
        return PPG_norm
    
    def PPG_normalization_V3(self, PPG):
        """
        :param PPG:
        :return: PPG_norm --> (-1, 1)
        """
        min_PPG = np.min(PPG)
        max_PPG = np.max(PPG)
        PPG_norm = PPG * 2.0 / (max_PPG - min_PPG)
        return PPG_norm
    
    def video_normalization(self, video_frames):
        diff_frames_np = np.array(video_frames)
        video_frames_scaled = np.zeros(shape=diff_frames_np.shape, dtype=np.float32)
        R_mean = np.mean(diff_frames_np[:, :, :, 0])
        G_mean = np.mean(diff_frames_np[:, :, :, 1])
        B_mean = np.mean(diff_frames_np[:, :, :, 2])
        
        R_std = np.std(diff_frames_np[:, :, :, 0])
        G_std = np.std(diff_frames_np[:, :, :, 1])
        B_std = np.std(diff_frames_np[:, :, :, 2])
        
        video_frames_scaled[:, :, :, 0] = (diff_frames_np[:, :, :, 0] - R_mean) / R_std
        video_frames_scaled[:, :, :, 1] = (diff_frames_np[:, :, :, 1] - G_mean) / G_std
        video_frames_scaled[:, :, :, 2] = (diff_frames_np[:, :, :, 2] - B_mean) / B_std
        
        return video_frames_scaled
    
    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        PPG = np.zeros((self.batch_size, *self.PPG_dim), dtype=np.float32)
        video = np.zeros((self.batch_size, 60, *self.img_dim), dtype=np.float32)
        HR = np.zeros((self.batch_size,), dtype=np.float32)
        
        x = np.linspace(0, 120, 120)
        x_new = np.linspace(0, 120, 60)
        
        # Generate batch data
        for i, filename in enumerate(list_files_temp):
            # print("filename: ", filename)
            path_npz = os.path.join(self.npz_dir, filename)
            data = np.load(path_npz)
            # video[i, ] = self.video_normalization(data["video"])
            agument_param = np.float32(random.uniform(-0.5, 0.5))
            video_temp = data["video"]
            video[i,] = video_temp + agument_param
            PPG_signal = data["PPG"]
            f_inter = interpolate.interp1d(x, PPG_signal, kind="cubic")
            PPG_inter = f_inter(x_new)
            PPG_filter = gaussian_filter1d(PPG_inter, 2)
            PPG_filter = self.PPG_normalization_V2(PPG_filter)
            PPG[i, ] = PPG_filter
            HR[i, ] = np.float32(data["HR"] - 1.2)
            
            # file_img_ppg = filename.replace(".npz", ".jpg")
            # path_img_ppg = os.path.join("G:/NBnpz_2021/train_Norm_npz_60/img_data_V2_10/", file_img_ppg)
            # img_ppg = cv2.imread(path_img_ppg)
            # cv2.imshow("PPG_img", img_ppg)
            # cv2.waitKey(1)
            #
            # plt.plot(PPG_signal[::2], label="original")
            # plt.plot(PPG_filter, label="filtered")
            # plt.legend()
            # plt.show()
        
        return video, [HR, PPG]


if __name__ == '__main__':
    # Parameters
    import random
    
    # npz_dir = "G:/NB_test_Norm_npz_60/npz_data_20/"
    npz_dir = "G:/NBnpz_2021/train_Norm_npz_60/npz_data_V2_10/"
    list_files = os.listdir(npz_dir)
    random.shuffle(list_files)
    
    data_generator = BatchPPGGenerator(npz_dir=npz_dir, list_files=list_files, batch_size=16)
    
    for i in range(200):
        video, PPG = data_generator.__getitem__(i)
        print("video shape:", video.shape)
        # print("PPG:", PPG)
        print("i: %d, finish one data batch!" % (i))
        for i in range(10):
            plt.plot(PPG[i, :])
            plt.show()

