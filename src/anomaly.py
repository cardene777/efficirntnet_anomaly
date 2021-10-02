import streamlit as st
import zipfile
from stqdm import stqdm

import os
from scipy import stats
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from sklearn import metrics
import seaborn as sns
from scipy.spatial import distance
import tensorflow as tf
import efficientnet_model


def image_transform(index, image_data, resize, train=""):
    image_data = cv2.imread(image_data)  # 画像の読み込み
    resize_image = cv2.resize(image_data, dsize=(resize, resize))  # 画像のリサイズ
    image_data_array = np.asarray(resize_image)  # numpy配列に変換
    image_data_array = image_data_array.astype('float32') / 255.0
    return image_data_array


def get_mean_cov(loader, features, level):
    feat = []
    # データを１つ１つ回す
    for inputs in stqdm(loader):
        feat_list = features[f"level_{level}"](tf.expand_dims(inputs, axis=0))
        feat_list = feat_list.numpy()
        for i in range(len(feat_list)):
            feat.append(feat_list[i].reshape(-1))

    feat = np.array(feat)  # numpy配列に変換
    mean = np.mean(feat, axis=0)  # 平均値
    cov = np.cov(feat.T)  # 共分散
    return feat, mean, cov


def get_score(feat, mean, cov):
    result = []
    cov_i = np.linalg.pinv(cov)
    for i in stqdm(range(len(feat))):
        result.append(distance.mahalanobis(feat[i], mean, cov_i))  # マハラノビス距離計算
    return result, cov_i


def get_auc(Z1, Z2):
    """
    Z1 : 検証用良品画像
    Z2 : 検証用不良画像
    """
    fig = plt.figure(figsize=(12, 8))
    plt.title("Mahalanobis distance")
    plt.plot(Z1, label="normal")
    plt.plot(Z2, label="anomaly")
    plt.legend()
    plt.savefig("Mahalanobis.png")
    st.pyplot(fig)

    # 散布図
    fig = plt.figure(figsize=(12, 8))
    plt.title("Mahalanobis distance scatter")
    plt.scatter(range(len(Z1)), Z1, label="normal", c="blue")
    plt.scatter(range(len(Z2)), Z2, label="anomaly", c="red")
    plt.legend()
    plt.savefig("Mahalanobis_scatter.png")
    st.pyplot(fig)

    y_true = np.zeros(len(Z1) + len(Z2))
    y_true[len(Z1):] = 1  # 0:正常、1：異常

    # FPR, TPR(, しきい値) を算出
    fpr, tpr, _ = metrics.roc_curve(y_true, np.hstack((Z1, Z2)))

    # AUC
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, auc


def plot_roc(Z1, Z2):
    '''
    Z1 : 検証用良品画像
    Z2 : 検証用不良画像
    '''
    fpr, tpr, auc1 = get_auc(Z1, Z2)
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 1, 1)
    plt.plot(fpr, tpr, label='(AUC = %.3f)' % auc1)
    plt.title('(ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig("ROC.png")
    st.pyplot(fig)

    return auc1, fpr, tpr


def hotelling_1d(data, threshold, kind, avg_var=None):
    # Covert raw data into the degree of abnormality
    if avg_var:
        avg = avg_var[0]
        var = avg_var[1]
    else:
        avg = np.average(data)
        var = np.var(data)
    data_abn = [(x - avg) ** 2 / var for x in data]

    abn_th = stats.chi2.interval(1 - threshold, 1)[1]

    # Set the threshold of abnormality
    if kind == "anomaly":
        fig = plt.figure(figsize=(12, 8))
        plt.title("Mahalanobis anomaly scatter")
        plt.scatter(range(len(data_abn)), data_abn, label="anomaly", c="blue")
        plt.hlines(abn_th, 0, len(data_abn), "red", linestyles='dashed', label=f"thr = {abn_th}")
        plt.legend()
        plt.savefig("anomaly_thr_scatter.png")
        st.pyplot(fig)
    else:
        fig = plt.figure(figsize=(12, 8))
        plt.title("Mahalanobis normal scatter")
        plt.scatter(range(len(data_abn)), data_abn, label="normal", c="blue")
        plt.hlines(abn_th, 0, len(data_abn), "red", linestyles='dashed', label=f"thr = {abn_th}")
        plt.legend()
        plt.savefig("normal_thr_scatter.png")
        st.pyplot(fig)

    # Abnormality determination
    result = []
    for (index, x) in enumerate(data_abn):
        if x > abn_th:
            result.append((index, data[index]))

    return result, avg, var


def get_score_tensorflow(feature, mean, cov_i):
    mean = tf.convert_to_tensor(mean, dtype=tf.float32)
    cov_i = tf.convert_to_tensor(cov_i, dtype=tf.float32)
    result = tf.tensordot(feature - mean, cov_i, axes=1)
    result = tf.tensordot(result, feature - mean, axes=1)
    result = tf.math.sqrt(result)
    return result


def make_fig(img):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:, :, i] = img[:, :, i] * img_std[i] + img_mean[i]
    return img


def inference_test(anomaly_array, features, mean, cov_i, alpha, lamda, level):
    figures = []
    mahalanobis = []

    sample_no = 1

    anomaly_image = anomaly_array[sample_no]

    figures.append(anomaly_image)

    anomaly_image = tf.Variable(anomaly_image)

    with tf.GradientTape() as tape:
        tape.watch(anomaly_image)
        feature = features[f"level_{level}"](tf.expand_dims(anomaly_image, axis=0))
        loss = get_score_tensorflow(tf.reshape(feature, -1), mean, cov_i)

    x_grad = tape.gradient(loss, anomaly_image)
    x_t = anomaly_image - alpha * x_grad

    # backward iteration
    st.text("=" * 30)
    for roop_num in range(30):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(anomaly_image)
            feature = features[f"level_{level}"](tf.expand_dims(x_t, axis=0))
            score = get_score_tensorflow(tf.reshape(feature, [-1]), mean, cov_i)
            loss = score + lamda * tf.reduce_sum(tf.math.abs(x_t - anomaly_image))

        mahalanobis_score = get_score_tensorflow(tf.reshape(feature, [-1]), mean, cov_i)
        if roop_num % 10 == 0:
            st.text(f"Mahalanobis distance {mahalanobis_score}")
        mahalanobis.append(mahalanobis_score)

        x_grad = tape2.gradient(loss, anomaly_image)
        x_t = x_t - alpha * x_grad
        figures.append(x_t.numpy())

    st.text("=" * 30)
    min_mahalanobis = min(mahalanobis)
    min_mahalanobis_index = mahalanobis.index(min_mahalanobis) + 1

    diff1 = np.abs(figures[0] - figures[min_mahalanobis_index])  # 不良画像と不良画像をできるだけ良品画像に近づけたものの差分の絶対値を取る。
    diff2 = np.sum(diff1, axis=-1)  # 合計を取得。
    diff3 = (diff2 - np.min(diff2)) / (np.max(diff2) - np.min(diff2))

    # ノイズ除去
    k_size = 5
    # 中央値フィルタ
    img_mask = cv2.medianBlur(diff3, k_size)

    img_gray = cv2.cvtColor(figures[0], cv2.COLOR_BGR2GRAY)

    jetcam = img_gray / 4 + img_mask   # もとの画像に合成

    fig = plt.figure(figsize=(12, 12))

    plt.subplot(1, 3, 1)
    plt.title(f"figures[0] socre - {min_mahalanobis}")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(figures[0], cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("img_mask")
    plt.axis("off")
    plt.imshow(img_mask)

    plt.subplot(1, 3, 3)
    plt.title(f"figures[{min_mahalanobis_index}]")
    plt.axis("off")
    plt.imshow(jetcam)

    st.pyplot(fig)


def inference(anomaly_array, features, mean, cov_i, alpha, lamda, level, number):
    mahalanobis_scores = []

    for number in range(number):
        figures = []
        mahalanobis = []

        anomaly_image = anomaly_array[number]

        figures.append(anomaly_image)

        anomaly_image = tf.Variable(anomaly_image)

        with tf.GradientTape() as tape:
            tape.watch(anomaly_image)
            feature = features[f"level_{level}"](tf.expand_dims(anomaly_image, axis=0))
            loss = get_score_tensorflow(tf.reshape(feature, -1), mean, cov_i)

        x_grad = tape.gradient(loss, anomaly_image)
        x_t = anomaly_image - alpha * x_grad

        # backward iteration
        for _ in range(20):
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(anomaly_image)
                feature = features[f"level_{level}"](tf.expand_dims(x_t, axis=0))
                score = get_score_tensorflow(tf.reshape(feature, [-1]), mean, cov_i)
                loss = score + lamda * tf.reduce_sum(tf.math.abs(x_t - anomaly_image))

            mahalanobis_score = get_score_tensorflow(tf.reshape(feature, [-1]), mean, cov_i)
            # st.text(f"Mahalanobis distance {mahalanobis_score}")
            mahalanobis.append(mahalanobis_score)

            x_grad = tape2.gradient(loss, anomaly_image)
            x_t = x_t - alpha * x_grad
            figures.append(x_t.numpy())

        min_mahalanobis = min(mahalanobis)
        min_mahalanobis_index = mahalanobis.index(min_mahalanobis) + 1

        diff1 = np.abs(figures[0] - figures[min_mahalanobis_index])  # 不良画像と不良画像をできるだけ良品画像に近づけたものの差分の絶対値を取る。
        diff2 = np.sum(diff1, axis=-1)  # 合計を取得。
        diff3 = (diff2 - np.min(diff2)) / (np.max(diff2) - np.min(diff2))

        # ノイズ除去
        k_size = 5
        # 中央値フィルタ
        img_mask = cv2.medianBlur(diff3, k_size)

        # 元画像をグレー画像に変換
        img_gray = cv2.cvtColor(figures[0], cv2.COLOR_BGR2GRAY)

        jetcam = img_gray / 4 + img_mask   # もとの画像に合成

        fig = plt.figure(figsize=(12, 12))

        plt.subplot(1, 3, 1)
        plt.title(f"figures[0] socre - {min_mahalanobis}")
        plt.axis("off")
        plt.imshow(cv2.cvtColor(figures[0], cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title("img_mask")
        plt.axis("off")
        plt.imshow(img_mask)

        plt.subplot(1, 3, 3)
        plt.title(f"figures[{min_mahalanobis_index}]")
        plt.axis("off")
        plt.imshow(jetcam)

        st.pyplot(fig)


def upload_images(train_image_dir, anomaly_image_dir, image_size):
    # 良品画像・不良品画像データのパス読み込み
    train_dir = glob.glob(os.path.join(train_image_dir, '*'))
    anomaly_dir = glob.glob(os.path.join(anomaly_image_dir, '*'))
    random.shuffle(anomaly_dir)

    normal_list = train_dir[len(train_dir) // 2:]
    train_list = train_dir[:len(train_dir) // 2]
    anomaly_list = anomaly_dir[:]

    # 画像変換
    train_array = []
    normal_array = []
    anomaly_array = []

    st.write("学習用良品画像前処理...")
    for index, image_path in enumerate(stqdm(train_list)):
        train_array.append(image_transform(index, image_path, image_size, train="train"))

    st.write("検証用良品画像前処理...")
    for index, image_path in enumerate(stqdm(normal_list)):
        normal_array.append(image_transform(index, image_path, image_size, train="normal"))

    st.write("不良品画像前処理...")
    for index, image_path in enumerate(stqdm(anomaly_list)):
        anomaly_array.append(image_transform(index, image_path, image_size))

    st.text(f"学習用良品画像{len(train_array)}枚")
    st.text(f"検証用良品画像{len(normal_array)}枚")
    st.text(f"不良品画像{len(anomaly_array)}枚")

    st.header("画像確認")
    st.image(cv2.cvtColor(train_array[0], cv2.COLOR_BGR2RGB), caption="学習用良品画像", channels='RGB')
    st.image(cv2.cvtColor(normal_array[0], cv2.COLOR_BGR2RGB), caption="検証用良品画像", channels='RGB')
    st.image(cv2.cvtColor(anomaly_array[0], cv2.COLOR_BGR2RGB), caption="不良品画像", channels='RGB')

    return train_array, normal_array, anomaly_array


def main():
    # 初期値設定
    MODEL = 5
    LEVEL = 7
    RESIZE = 456
    APLHA = 0.0001
    LAMDA = 1

    train_image_dir = "good"  # 学習用良品画像データ
    anomaly_image_dir = "anomaly"  # 不良画像

    KERNEL = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # シード値を設定
    def set_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    set_seed(42)

    st.title("EfficientNet")

    # 良品画像と不良品画像が入ったzipファイルをアップロード
    uploaded_train_zip_file = st.file_uploader('upload train zip file')

    if uploaded_train_zip_file:
        try:
            with zipfile.ZipFile(uploaded_train_zip_file) as train_zip:
                train_zip.extractall()
        except:
            st.text("zipファイルをアップロードしてください。")
    else:
        st.text("アップロードされていません。")

    uploaded_anomaly_zip_file = st.file_uploader('upload anomaly zip file')
    if uploaded_anomaly_zip_file:
        try:
            with zipfile.ZipFile(uploaded_anomaly_zip_file) as anomaly_zip:
                anomaly_zip.extractall()
        except:
            st.text("zipファイルをアップロードしてください。")
    else:
        st.text("アップロードされていません。")

    if uploaded_train_zip_file and uploaded_anomaly_zip_file:
        # 画像の前処理
        train_array, normal_array, anomaly_array = upload_images(train_image_dir, anomaly_image_dir, RESIZE)

        # モデルをダウンロード
        st.write("モデルをダウンロード...")
        inputs = tf.keras.layers.Input(shape=(RESIZE, RESIZE, 3))
        model, features = efficientnet_model.EfficientNetB5(include_top=False, input_tensor=inputs)
        st.write("モデルのダウンロード完了！")

        # モデルに通す
        st.write("学習用良品画像をモデルに通す...")
        train_feat, mean, cov = get_mean_cov(train_array, features, LEVEL)  # numpy配列, 平均値, 共分散
        st.write("検証用良品画像をモデルに通す...")
        normal_feat, _, _ = get_mean_cov(normal_array, features, LEVEL)
        st.write("不良品画像をモデルに通す...")
        anomaly_feat, _, _ = get_mean_cov(anomaly_array, features, LEVEL)
        st.write("全データをモデルに通せました！")

        # マハラノビス距離計算
        st.write("学習用良品画像のマハラノビス距離を計算...")
        train_score, _ = get_score(train_feat, mean, cov)  # 検証用良品画像
        st.write("検証用良品画像のマハラノビス距離を計算...")
        normal_score, cov_i = get_score(normal_feat, mean, cov)  # 検証用良品画像
        st.write("不良品画像のマハラノビス距離を計算...")
        anomaly_score, _ = get_score(anomaly_feat, mean, cov)  # 検証用不良画像
        st.write("全データのマハラノビス距離を計算できました！")

        fig = plt.figure(figsize=(12, 8))
        plt.plot(train_score, label="train")
        plt.legend()
        st.pyplot(fig, caption="学習用良品画像データのマハラノビス距離分布")

        fig = plt.figure(figsize=(12, 8))
        plt.plot(normal_score, label="normal")
        plt.legend()
        st.pyplot(fig, caption="検証用良品画像データのマハラノビス距離分布")

        fig = plt.figure(figsize=(12, 8))
        plt.plot(anomaly_score, label="anomaly")
        plt.legend()
        st.pyplot(fig, caption="不良品画像データのマハラノビス距離分布")

        # ROC計算
        _, f_pr, t_pr = plot_roc(normal_score, anomaly_score)

        # 異常検知（ホテリング理論）
        normal_result, avg, var = hotelling_1d(normal_score, 0.01, "normal")
        anomaly_result, _, _ = hotelling_1d(anomaly_score, 0.01, "anomaly", [avg, var])
        st.text(f"良品画像{len(normal_score)}枚の内、{len(normal_result)}枚が異常と判定されてしまった。")
        st.text(f"不良品画像{len(anomaly_score)}枚の内、{len(anomaly_score) - len(anomaly_result)}枚が正常と判定されてしまった。")

        # 混同行列作成
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap([[len(anomaly_result), len(anomaly_score) - len(anomaly_result)],
                     [len(normal_result), len(normal_score) - len(normal_result)]],
                    annot=True, cmap='Blues', fmt="d", cbar=False)
        plt.ylabel("anomaly（１）・good（０）")
        plt.xlabel("anomaly（１）・good（０）")
        plt.savefig('confusion_matrix.png')
        st.pyplot(fig)

        # 推論（テスト）
        inference_test(anomaly_array, features, mean, cov_i, APLHA, LAMDA, LEVEL)

        # 推論（指定枚数分）
        number = st.slider('何枚の不良品画像の異常部分を可視化しますか？', 0, len(anomaly_score), 10)
        # start_btn = st.button('実行')
        # if start_btn:
        if number > 0:
            inference(anomaly_array, features, mean, cov_i, APLHA, LAMDA, LEVEL, number)
        else:
            st.text("スライドバーで可視化したい枚数を指定してください。")


if __name__ == '__main__':
    main()
