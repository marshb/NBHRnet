import os
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import keras.backend as K
import random

from PearsonCorr import pearson_r, pearson_mse
from batchDataGenerator_fusemodel import BatchPPGGenerator
from NN_models import NNmodels


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def scheduler(epoch):
    # 学习率减小为原来的1/3
    print("current learning rate:", K.get_value(model_fusion.optimizer.lr))
    if epoch % 30 == 0 and epoch != 0:
        lr = K.get_value(model_fusion.optimizer.lr)
        K.set_value(model_fusion.optimizer.lr, lr * 0.2)
        print("lr changed to {}".format(lr * 0.2))
    return K.get_value(model_fusion.optimizer.lr)


if __name__ == "__main__":
    # train_rate = 0.95
    batch_size = 12
    save_model_period = 10
    lr = 1e-4
    NUM_EPOCH = 50
    
    train_npz_dir = "train_data_directory"
    test_npz_dir = "test_data_directory"
    
    train_list_files = os.listdir(train_npz_dir)
    random.shuffle(train_list_files)
    
    count_train = len(train_list_files)
    print("####  The total number of training samples: ", len(train_list_files))
    test_list_files = os.listdir(test_npz_dir)
    print("####  The total number of testing samples: ", len(test_list_files))
    
    train_generator = BatchPPGGenerator(npz_dir=train_npz_dir, list_files=train_list_files, batch_size=batch_size)
    test_generator = BatchPPGGenerator(npz_dir=test_npz_dir, list_files=test_list_files, batch_size=batch_size)
    
    steps_per_epoch = int(count_train / batch_size)
    
    # todo: my designed model
    model_class = NNmodels()
    model_fusion = model_class.PPG_extractor_model()
    model_name = "PPG_extractor_model"
    model_fusion.summary()
    
    path_PPG_model = "path_PPG_model"
    path_HR_model = "path_HR_model"
    model_fusion.load_weights(path_PPG_model, by_name=True)
    print("### loaded PPG model !")
    model_fusion.load_weights(path_HR_model, by_name=True)
    print("### loaded HR model !")
    
    model_dir_base = os.path.join("./models", "newborn")
    model_dir = os.path.join(model_dir_base, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("****** Create model dir:", model_dir)
    
    log_dir_base = os.path.join("./logs", "newborn")
    log_dir = os.path.join(log_dir_base, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print("****** Create log dir:", log_dir)
    
    losses = {
        "PPG_out": pearson_mse,
        "HR_out": "mse",
    }
    lossWeights = {"PPG_out": 0.1, "HR_out": 0.9}
    model_fusion.compile(loss=losses, loss_weights=lossWeights, optimizer=RMSprop(lr=lr), metrics=["mse"])
    
    reduce_lr = LearningRateScheduler(scheduler)
    checkpoint_path = os.path.join(model_dir, 'model_ep{epoch:03d}.h5')
    
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 verbose=1,
                                 monitor='val_loss',
                                 save_weights_only=False,
                                 period=save_model_period)  # Save model, every 5-epochs.
    
    print("model name: ", model_name)
    history = model_fusion.fit_generator(generator=train_generator,
                                      epochs=NUM_EPOCH,
                                      steps_per_epoch=steps_per_epoch,
                                      shuffle=True,
                                      validation_data=test_generator,
                                      # use_multiprocessing=False,
                                      max_queue_size=64,
                                      workers=4,
                                      verbose=1,
                                      initial_epoch=0,
                                      callbacks=[reduce_lr, checkpoint,
                                                 TensorBoard(log_dir=log_dir)])
    
    model_name_saved = model_name + '_ep' + str(NUM_EPOCH) + '_final' + '.h5'
    path_model_saved = os.path.join(model_dir, model_name_saved)
    model_fusion.save(path_model_saved)
    print("saved model:", path_model_saved)

