from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = 1 - r_num / r_den
    return K.mean(r)


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def pearson_mse(y_true, y_pred, theta=0.8):
    pearson_loss = pearson_r(y_true, y_pred)
    mse_loss = mse(y_true, y_pred)
    return theta * pearson_loss + (1.0 - theta) * mse_loss


def PPMC(predict, real):  # Pearson Product Moment Correlation, PPMC
    predict_mean = np.mean(real)
    real_mean = np.mean(predict)

    numerator = np.sum((predict - predict_mean) * (real - real_mean))
    denominator = np.sqrt(np.sum(np.square(predict - predict_mean))) * np.sqrt(
        np.sum(np.square(real - real_mean)))
    PPMC = numerator / denominator
    return PPMC


def sin_function(fs, A, phi):
    stop = 2.0*fs
    dt = np.linspace(start=0, stop=stop, num=120, dtype=np.float32)
    signal = A * np.sin(2.0 * np.pi * dt + phi)
    return signal


if __name__ == "__main__":
    fs_gt = 1.5
    pred_fs = 1.5
    phi_gt = 0.5
    phi_pred = 1.2
    A_gt = 1
    A_pred = 1
    signal_1 = sin_function(fs_gt, A_gt, phi_gt)
    signal_2 = sin_function(pred_fs, A_pred, phi_pred)
    ppmc = PPMC(signal_1, signal_2)
    ppmc = np.float16(ppmc)
    plt.plot(signal_1, label="ground truth")
    plt.plot(signal_2, label="predict")
    plt.legend(loc=1)
    plt.title("GT φ: {:.2f}, Pred φ: {:0.2f}, PPMC: {:.3f}".format(phi_gt, phi_pred, ppmc))
    plt.show()

    print("PPMC:", ppmc)

