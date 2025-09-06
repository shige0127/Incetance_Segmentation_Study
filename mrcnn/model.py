"""
Mask R-CNN
メインのMask R-CNNモデル実装。

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
from keras import layers as KE
import keras.models as KM

from mrcnn import utils

# TensorFlow 1.3+ と Keras 2.0.8+ が必要
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  ユーティリティ関数
############################################################

def log(text, array=None):
    """テキストメッセージを印刷します。オプションでNumpy配列が提供された場合、
    その形状、最小値、最大値を印刷します。
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Keras BatchNormalizationクラスを拡張して、必要に応じて変更を
    一元的に行えるようにします。

    バッチサイズが小さい場合、バッチ正規化は訓練に悪影響を与えるため、
    このレイヤーはしばしば凍結され（Configクラスの設定で）、
    線形レイヤーとして機能します。
    """
    def call(self, inputs, training=None):
        """
        training値に関する注意：
            None: BNレイヤーを訓練します。これは通常のモード
            False: BNレイヤーを凍結します。バッチサイズが小さい場合に適している
            True: （使用しない）推論時でもレイヤーを訓練モードに設定
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """バックボーンネットワークの各ステージの幅と高さを計算します。

    Returns:
        [N, (height, width)]. Nはステージ数
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # 現在はResNetのみサポート
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            # BACKBONE_STRIDES = [4, 8, 16, 32, 64]
            for stride in config.BACKBONE_STRIDES])


############################################################
#  ResNet グラフ
############################################################

# 以下からコードを採用:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """identity_blockはショートカットにconv層を持たないブロックです
    # 引数
        input_tensor: 入力テンソル
        kernel_size: デフォルト3、メインパスの中間conv層のカーネルサイズ
        filters: 整数のリスト、メインパスの3つのconv層のフィルタ数
        stage: 整数、現在のステージラベル、レイヤー名生成に使用
        block: 'a','b'..., 現在のブロックラベル、レイヤー名生成に使用
        use_bias: Boolean。conv層でバイアスを使用するかどうか
        train_bn: Boolean。Batch Norm層を訓練するか凍結するか
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # バイアスは線形変換の切片のこと
    # 勾配消失の問題を解決できる
    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)
    # ショートカット、ここでinput_tensorが入る
    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_blockはショートカットにconv層を持つブロックです
    # 引数
        input_tensor: 入力テンソル
        kernel_size: デフォルト3、メインパスの中間conv層のカーネルサイズ
        filters: 整数のリスト、メインパスの3つのconv層のフィルタ数
        stage: 整数、現在のステージラベル、レイヤー名生成に使用
        block: 'a','b'..., 現在のブロックラベル、レイヤー名生成に使用
        use_bias: Boolean。conv層でバイアスを使用するかどうか
        train_bn: Boolean。Batch Norm層を訓練するか凍結するか
    注意：stage 3以降、メインパスの最初のconv層はsubsample=(2,2)になります
    そしてショートカットもsubsample=(2,2)である必要があります
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """ResNetグラフを構築します。
        architecture: resnet50またはresnet101を指定可能
        stage5: Boolean。Falseの場合、ネットワークのstage5は作成されません
        train_bn: Boolean。Batch Norm層を訓練するか凍結するか
    """
    # 引数architectureがresnet50またはresnet101であることを確認
    assert architecture in ["resnet50", "resnet101"]
    # ステージ 1
    # 入力画像に対して上下左右に3ピクセルずつゼロパディングを追加
    x = KL.ZeroPadding2D((3, 3))(input_image)
    # 7x7の畳み込み層、64チャネル出力、ストライド2で画像サイズを半分に
    # ストライド2で位置を飛ばして計算するため小さくなる
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    # バッチ正規化層を適用（train_bnパラメータで学習/凍結を制御）
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    # ReLU活性化関数を適用
    x = KL.Activation('relu')(x)
    # 3x3の最大プーリング、ストライド2でさらに画像サイズを半分に、C1として保存
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # ステージ 2
    # conv_block: ダウンサンプリング付きのResNetブロック、チャネル数を64→256に増加
    # x,3の3は3x3のカーネルサイズを指定
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    # identity_block: スキップ接続付きのResNetブロック、チャネル数維持
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    # 3つ目のidentity_blockの出力をC2として保存（FPNで使用）
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # ステージ 3
    # conv_blockでダウンサンプリング、チャネル数を256→512に増加
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    # 3つのidentity_blockを連続適用、チャネル数512を維持
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    # 4つ目のidentity_blockの出力をC3として保存（FPNで使用）
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # ステージ 4
    # conv_blockでダウンサンプリング、チャネル数を512→1024に増加
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    # ResNet50では5個、ResNet101では22個のidentity_blockを繰り返し
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    # 各identity_blockにはb,c,d,e...のブロック名を付与（chr(98)='b'から開始）
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    # ステージ4の最終出力をC4として保存
    C4 = x
    # ステージ 5
    # stage5フラグがTrueの場合のみステージ5を構築
    if stage5:
        # conv_blockでダウンサンプリング、チャネル数を1024→2048に増加
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        # 2つのidentity_blockを連続適用
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        # 最終identity_blockの出力をC5として保存
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        # stage5を使用しない場合はC5をNoneに設定
        C5 = None
    # 各ステージの出力特徴マップをリストで返す（FPNで使用）
    return [C1, C2, C3, C4, C5]


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """与えられたデルタを与えられたボックスに適用します。
    boxes: [N, (y1, x1, y2, x2)] 更新するボックス
    deltas: [N, (dy, dx, log(dh), log(dw))] 適用する調整値
    """
    # y, x, h, w に変換
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # デルタを適用
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.math.exp(deltas[:, 2])
    width *= tf.math.exp(deltas[:, 3])
    # y1, x1, y2, x2 に逆変換
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] y1, x1, y2, x2の形式
    """
    # 分割
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # クリップ
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(KE.Layer):
    """アンカースコアを受け取り、第2段階に提案として渡すサブセットを選択します。
    フィルタリングはアンカースコアと重複を除去するnon-max suppressionに基づいて行われます。
    また、アンカーにバウンディングボックス調整デルタを適用します。

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] 正規化座標のアンカー

    Returns:
        正規化座標の提案 [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # ボックススコア。前景クラス信頼度を使用。[Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # ボックスデルタ deltas.shape = [batch_size, num_anchors, 4]
        # アンカーから正解への調整量
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # アンカー
        anchors = inputs[2]

        # スコア順に上位アンカーに絞ってパフォーマンスを向上させ
        # より小さなサブセットで残りの処理を行う。
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        # サイズ1のバッチにする、一部のカスタムレイヤーはバッチサイズ1のみをサポート
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # デルタをアンカーに適用して精密化されたアンカーを取得。
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # 画像境界にクリップ。正規化座標のため、0..1の範囲にクリップ。
        # [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # 小さなボックスをフィルタリング
        # Xinlei Chenの論文によると、これは小さなオブジェクトの検出精度を
        # 低下させるため、スキップする。

        # 非最大抑制
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # 必要に応じてパディング
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


############################################################
#  ROIAlign層
############################################################

def log2_graph(x):
    """Log2の実装。TensorFlowにはネイティブ実装がない。"""
    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KE.Layer):
    """特徴ピラミッドの複数レベルでROI Poolingを実装する。

    パラメータ:
    - pool_shape: 出力プール領域の[pool_height, pool_width]。通常[7, 7]

    入力:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] 正規化座標。
             配列を埋めるのに十分なボックスがない場合はゼロでパディングされることがある。
    - image_meta: [batch, (メタデータ)] 画像の詳細。compose_image_meta()を参照
    - feature_maps: ピラミッドの異なるレベルからの特徴マップのリスト。
                    各々は[batch, height, width, channels]

    出力:
    [batch, num_boxes, pool_height, pool_width, channels]形状のプール領域。
    幅と高さはレイヤーコンストラクタのpool_shapeで指定されたもの。
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # 正規化座標でのボックスをクロップ [batch, num_boxes, (y1, x1, y2, x2)]
        boxes = inputs[0]

        # 画像メタデータ
        # 画像の詳細情報を保持。compose_image_meta()を参照
        image_meta = inputs[1]

        # 特徴マップ。特徴ピラミッドの異なるレベルからの特徴マップのリスト。
        # 各々は[batch, height, width, channels]
        feature_maps = inputs[2:]

        # ROIの面積に基づいて各ROIをピラミッドのレベルに割り当てる。
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # 最初の画像の形状を使用。バッチ内の画像は同じサイズである必要がある。
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Feature Pyramid Networks論文の式1。ここでは座標が正規化されている
        # ことを考慮。例: 224x224 ROI（ピクセル）はP4にマップされる
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # レベルをループして各々にROI poolingを適用。P2からP5まで。
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # crop_and_resize用のボックスインデックス。
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # どのボックスがどのレベルにマップされるかを追跡
            box_to_level.append(ix)

            # ROI提案への勾配伝播を停止
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # クロップおよびリサイズ
            # Mask R-CNN論文より: 「4つの正規位置をサンプルして、
            # max poolingまたはaverage poolingのいずれかを評価できるようにする。
            # 実際、各ビンの中心で単一の値を補間するだけでも
            # （poolingなしで）ほぼ同等に効果的である。」
            #
            # ここではビン当たり単一値の簡化されたアプローチを使用。
            # これはtf.crop_and_resize()で行われる方法。
            # 結果: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # プールされた特徴を1つのテンソルにパック
        pooled = tf.concat(pooled, axis=0)

        # box_to_levelマッピングを1つの配列にパックし、
        # プールされたボックスの順序を表す列を追加
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # 元のボックスの順序に合わせてプールされた特徴を再配置
        # box_to_levelをバッチ、次にボックスインデックスでソート
        # TFには2列でソートする方法がないため、マージしてソート。
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # バッチ次元を再追加
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


############################################################
#  検出ターゲット層
############################################################

def overlaps_graph(boxes1, boxes2):
    """二つのボックスセット間のIoU重なりを計算する。
    boxes1, boxes2: [N, (y1, x1, y2, x2)]。
    """
    # 1. boxes2をタイル化しboxes1を繰り返す。これにより比較可能
    # ループなしで全てのboxes1を全てのboxes2に対して比較。
    # TFにnp.repeat()の相当品がないためシミュレート
    # tf.tile()とtf.reshape()を使用。
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. 交差を計算
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. 合併を計算
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. IoUを計算して[boxes1, boxes2]に再整形
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """一枚の画像に対する検出ターゲットを生成する。提案をサブサンプルし、
    各々のターゲットクラスID、バウンディングボックスデルタ、マスクを生成する。

    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] 正規化座標。提案が不足している場合は
               ゼロパディングされることがある。
    gt_class_ids: [MAX_GT_INSTANCES] 整数型クラスID
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] 正規化座標。
    gt_masks: [height, width, MAX_GT_INSTANCES] ブール型。

    Returns: ターゲットROIと対応するクラスID、バウンディングボックスシフト、
    およびマスク。
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] 正規化座標
    class_ids: [TRAIN_ROIS_PER_IMAGE]。整数型クラスID。ゼロパディング。
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]。bbox境界にクロップされ、
           ニューラルネットワーク出力サイズにリサイズされたマスク。

    注意：ターゲットROIが不足している場合、返される配列はゼロパディングされることがある。
    """
    # アサーション
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # ゼロパディングを削除
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # COCOの群衆を処理
    # COCOの群衆ボックスは複数のインスタンスを囲むバウンディングボックス。
    # 訓練から除外する。群衆ボックスには負のクラスIDが付与される。
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # 重なり行列を計算 [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # 群衆ボックスとの重なりを計算 [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # ポジティブおよびネガティブROIを決定
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. ポジティブROIはGTボックスとのIoU >= 0.5のもの
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. ネガティブROIは全てのGTボックスとのIoU < 0.5のもの。群衆をスキップ。
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # ROIをサブサンプル。ポジティブ33%を目指す
    # ポジティブROI
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # ネガティブROI。ポジティブ:ネガティブの比率を維持するのに十分な数を追加。
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # 選択されたROIを収集
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # ポジティブROIをGTボックスに割り当てる。
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # ポジティブROIのbbox精密化を計算
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # ポジティブROIをGTマスクに割り当て
    # マスクを[N, height, width, 1]に並べ替え
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # 各ROIに対して正しいマスクを選択
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # マスクターゲットを計算
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # ROI座標を正規化画像空間から変換
        # 正規化されたミニマスク空間へ。
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # マスクから余分な次元を削除。
    masks = tf.squeeze(masks, axis=3)

    # バイナリクロスエントロピー损失で使用するため、
    # GTマスクを0または1にするためマスクピクセルを0.5で闾値処理。
    masks = tf.round(masks)

    # ネガティブROIを追加し、ネガティブROIに使用されない
    # bboxデルタとマスクをゼロでパディング。
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """提案をサブサンプルし、各々に対してターゲットボックス精密化、class_ids、
    およびマスクを生成する。

    入力:
    proposals: [batch, N, (y1, x1, y2, x2)] 正規化座標。提案が不足している場合は
               ゼロパディングされることがある。
    gt_class_ids: [batch, MAX_GT_INSTANCES] 整数型クラスID。
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] 正規化座標。
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] ブール型

    返し値: ターゲットROIと対応するクラスID、バウンディングボックスシフト、
    およびマスク。
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] 正規化座標
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]。整数型クラスID。
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 bbox境界にクロップされ、ニューラルネットワーク出力サイズに
                 リサイズされたマスク。

    注意: ターゲットROIが不足している場合、返される配列はゼロパディングされることがある。
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # バッチをスライスし、各スライスに対してグラフを実行
        # TODO: 明確化のためtarget_bboxをtarget_deltasにリネーム
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


############################################################
#  検出層
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """分類された提案を精密化し、重なりをフィルタリングして最終検出結果を返す。

    入力:
        rois: [N, (y1, x1, y2, x2)] 正規化座標
        probs: [N, num_classes]。クラス確率。
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]。クラス固有の
                バウンディングボックスデルタ。
        window: (y1, x1, y2, x2) 正規化座標。パディングを除いた画像を含む
            画像の部分。

    返し値の形状: [num_detections, (y1, x1, y2, x2, class_id, score)]、座標は正規化。
    """
    # ROIごとのクラスID
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # 各ROIのトップクラスのクラス確率
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # クラス固有のバウンディングボックスデルタ
    deltas_specific = tf.gather_nd(deltas, indices)
    # バウンディングボックスデルタを適用
    # 形状: [boxes, (y1, x1, y2, x2)] 正規化座標
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # ボックスを画像ウィンドウにクリップ
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: 面積ゼロのボックスをフィルタリング

    # 背景ボックスをフィルタリング
    keep = tf.where(class_ids > 0)[:, 0]
    # 低信頼度ボックスをフィルタリング
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # クラスごとのNMSを適用
    # 1. 変数を準備
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """指定されたクラスのROIに非最大抑制を適用する。"""
        # 指定されたクラスのROIのインデックス
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # NMSを適用
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # インデックスをマップ
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # 返されるテンソルが同じ形状になるように-1でパディング
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # map_fn()が結果の形状を推定できるように形状を設定
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. クラスIDをマップ
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. 結果を1つのリストにマージし、-1のパディングを削除
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. keepとnms_keepの交差を計算
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # トップ検出を保持
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # 出力を[N, (y1, x1, y2, x2, class_id, score)]に配置
    # 座標は正規化されている。
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # detections < DETECTION_MAX_INSTANCESの場合ゼロでパディング
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KE.Layer):
    """分類された提案ボックスとそのバウンディングボックスデルタを受け取り、
    最終的な検出ボックスを返す。

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]
    座標は正規化済み。
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # 正規化座標での画像ウィンドウを取得。ウィンドウは
        # パディングを除いた画像内の領域。
        # バッチ内の最初の画像の形状を使用してウィンドウを正規化
        # 全ての画像が同じサイズにリサイズされることが分かっているため。
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # バッチ内の各アイテムに対して検出精密化グラフを実行
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # 出力を再整形
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]
        # 正規化座標
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  領域提案ネットワーク (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """領域提案ネットワークの計算グラフを構築する。

    feature_map: バックボーンの特徴 [batch, height, width, depth]
    anchors_per_location: 特徴マップ内のピクセル毎のアンカー数
    anchor_stride: アンカーの密度を制御。通常は1（特徴マップの全ピクセル）
                   または2（1ピクセル置き）。

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] アンカー分類器のlogits（softmax前）
        rpn_probs: [batch, H * W * anchors_per_location, 2] アンカー分類器の確率。
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] アンカーに
                  適用される差分値。
    """
    # TODO: 特徴マップが偶数でない場合、stride=2がアライメントの問題を引き起こすか確認。
    # RPNの共有畳み込みベース
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # アンカースコア。[batch, height, width, anchors per location * 2]。
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # [batch, anchors, 2]に再整形
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]),
        output_shape=lambda s: (s[0], None, 2))(x)

    # BG/FGの最後の次元にSoftmaxを適用。
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # バウンディングボックスの精密化。[batch, H, W, anchors per location * depth]
    # depthは[x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # [batch, anchors, 4]に再整形
    rpn_bbox = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]),
        output_shape=lambda s: (s[0], None, 4))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """領域提案ネットワークのKerasモデルを構築する。
    RPNグラフをラップして、共有重みで繰り返し使用できるようにする。

    anchors_per_location: 特徴マップ内のピクセル毎のアンカー数
    anchor_stride: アンカーの密度を制御。通常は1（特徴マップの全ピクセル）
                   または2（1ピクセル置き）。
    depth: バックボーン特徴マップの深さ。

    Keras Modelオブジェクトを返す。モデルの出力は呼び出された時：
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] アンカー分類器のlogits（softmax前）
    rpn_probs: [batch, H * W * anchors_per_location, 2] アンカー分類器の確率。
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] アンカーに
                適用される差分値。
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  特徴ピラミッドネットワークヘッド
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    """特徴ピラミッドネットワークの分類器および回帰ヘッドの計算グラフを構築する。

    rois: [batch, num_rois, (y1, x1, y2, x2)] 正規化座標での提案ボックス。
    feature_maps: ピラミッドの異なるレイヤーからの特徴マップのリスト、
                  [P2, P3, P4, P5]。各々異なる解像度。
    image_meta: [batch, (meta data)] 画像の詳細情報。compose_image_meta()を参照
    pool_size: ROI Poolingで生成される正方形特徴マップの幅。
    num_classes: 結果の深さを決めるクラス数
    train_bn: ブール値。Batch Normレイヤーの訓練または固定
    fc_layers_size: 2つのFCレイヤーのサイズ

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] 分類器のlogits（softmax前）
        probs: [batch, num_rois, NUM_CLASSES] 分類器の確率
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] 提案ボックスに
                     適用される差分値
    """
    # ROIプーリング
    # 形状: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # 2つの1024 FCレイヤー（一貫性のためConv2Dで実装）
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       output_shape=lambda s: (s[0], s[1]),
                       name="pool_squeeze")(x)

    # 分類器ヘッド
    mrcnn_class_logits = KL.Dense(num_classes, name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.Activation("softmax", name="mrcnn_class")(mrcnn_class_logits)

    # バウンディングボックスヘッド
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.Dense(num_classes * 4, activation='linear', name='mrcnn_bbox_fc')(shared)
    # [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]に再整形
    # TensorFlow 2.xとの互換性向上のため動的再整形を使用
    mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """特徴ピラミッドネットワークのマスクヘッドの計算グラフを構築する。

    rois: [batch, num_rois, (y1, x1, y2, x2)] 正規化座標での提案ボックス。
    feature_maps: ピラミッドの異なるレイヤーからの特徴マップのリスト、
                  [P2, P3, P4, P5]。各々異なる解像度。
    image_meta: [batch, (meta data)] 画像の詳細情報。compose_image_meta()を参照
    pool_size: ROI Poolingで生成される正方形特徴マップの幅。
    num_classes: 結果の深さを決めるクラス数
    train_bn: ブール値。Batch Normレイヤーの訓練または固定

    Returns: マスク [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROIプーリング
    # 形状: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # 畳み込みレイヤー
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x


############################################################
#  損失関数
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Smooth-L1损失を実装する。
    y_trueおy_predは通常: [N, 4]、しかし任意の形状可能。
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPNアンカー分類器损失。

    rpn_match: [batch, anchors, 1]。アンカーマッチタイプ。1=positive,
               -1=negative, 0=neutral anchor。
    rpn_class_logits: [batch, anchors, 2]。BG/FG用のRPN分類器logits。
    """
    # 簡化のため最後の次元を圧縮
    rpn_match = tf.squeeze(rpn_match, -1)
    # アンカークラスを取得。-1/+1マッチを0/1値に変換。
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # ポジティブとネガティブアンカーは损失に貢献するが、
    # 中立アンカー（マッチ値 = 0）は貢献しない。
    indices = tf.where(K.not_equal(rpn_match, 0))
    # 损失に貢献する行を選び、他をフィルタリング。
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # クロスエントロピー损失
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """RPNバウンディングボックス损失グラフを返す。

    config: モデル設定オブジェクト。
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))]。
        未使用のbboxデルタを埋めるために0パディングを使用。
    rpn_match: [batch, anchors, 1]。アンカーマッチタイプ。1=positive,
               -1=negative, 0=neutral anchor。
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # ポジティブアンカーは损失に貢献するが、ネガティブと
    # 中立アンカー（マッチ値が0または-1）は貢献しない。
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # 损失に貢献するbboxデルタを選択
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # ターゲットバウンディングボックスデルタをrpn_bboxと同じ長さにトリム。
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)
    
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    """Mask RCNNの分類器ヘッドの損失。

    target_class_ids: [batch, num_rois]. 整数クラスID。配列を埋めるために
        ゼロパディングを使用。
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. 画像のデータセットに含まれる
        クラスには1の値、データセットにないクラスには0の値。
    """
    # モデル構築中、Kerasはこの関数をfloat32型のtarget_class_idsで呼び出す。
    # 理由は不明。回避するためintにキャストする。
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # データセットにないクラスの予測を検出。
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: バッチ > 1で動作するようにこの行を更新。現在はバッチ内の全ての
    #       画像が同じactive_class_idsを持つことを仮定している。
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # 損失
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # 画像のアクティブクラスに含まれないクラスの予測の損失を消去。
    loss = loss * pred_active

    # 損失の平均を計算。正しい平均を得るため、損失に貢献する予測のみ使用。
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Mask R-CNNバウンディングボックス精細化の損失。

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. 整数クラスID。
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # 簡化のためバッチとroi次元をマージして再整形。
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))

    # ポジティブROIのみ損失に貢献。そして各ROIの正しいclass_idのみ。
    # それらのインデックスを取得。
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # 損失に貢献するデルタ（予測値と真値）を収集
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # スムーズL1損失
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """マスクヘッドのマスクバイナリクロスエントロピー損失。

    target_masks: [batch, num_rois, height, width].
        0または1の値のfloat32テンソル。配列を埋めるためゼロパディングを使用。
    target_class_ids: [batch, num_rois]. 整数クラスID。ゼロパディング済み。
    pred_masks: [batch, proposals, height, width, num_classes] 0から1の値の
                float32テンソル。
    """
    # 簡化のため再整形。最初の2つの次元をマージ。
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # 予測マスクを[N, num_classes, height, width]に置換
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # ポジティブROIのみ損失に貢献。そして各ROIのクラス固有マスクのみ。
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # 損失に貢献するマスク（予測値と真値）を収集
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # バイナリクロスエントロピーを計算。ポジティブROIがない場合は0を返す。
    # 形状: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


############################################################
#  データジェネレーター
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """画像の正解データ（画像、マスク、バウンディングボックス）を読み込んで返す。

    augment: （非推奨。代わりにaugmentationを使用）。Trueの場合、ランダムな
        画像拡張を適用。現在は水平反転のみ提供。
    augmentation: オプション。imgaug (https://github.com/aleju/imgaug) 拡張。
        例：imgaug.augmenters.Fliplr(0.5)を渡すと50%の確率で画像を
        左右反転させる。
    use_mini_mask: Falseの場合、元画像と同じ高さと幅のフルサイズマスクを返す。
        これらは大きくなる可能性があり、例えば1024x1024x100（100インスタンス）。
        ミニマスクはより小さく、通常224x224で、オブジェクトのバウンディングボックスを
        抽出してMINI_MASK_SHAPEにリサイズすることで生成される。

    Returns:
    image: [height, width, 3]
    shape: リサイズとクロッピング前の画像の元の形状。
    class_ids: [instance_count] 整数クラスID
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. 高さと幅は画像のものだが、
        use_mini_maskがTrueの場合はMINI_MASK_SHAPEで定義される。
    """
    # 画像とマスクを読み込み
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # ランダムな水平反転。
    # TODO: 将来のアップデートでaugmentationに置き換えられて削除予定
    if augment:
        logging.warning("'augment'は非推奨です。代わりに'augmentation'を使用してください。")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # データ拡張
    # imgaugライブラリが必要 (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # マスクに安全に適用できる拡張器
        # Affineなど一部の拡張器には安全でない設定があるため、
        # 必ずマスクで拡張をテストする
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """マスクに適用する拡張器を決定する。"""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # 比較のため拡張前の形状を保存
        image_shape = image.shape
        mask_shape = mask.shape
        # 画像とマスクに同様に適用するため拡張器を決定的にする
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # imgaugがnp.boolをサポートしないためマスクをnp.uint8に変更
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # 形状が変わっていないことを確認
        assert image.shape == image_shape, "データ拡張は画像サイズを変更すべきではありません"
        assert mask.shape == mask_shape, "データ拡張はマスクサイズを変更すべきではありません"
        # マスクをboolに戻す
        mask = mask.astype(np.bool)

    # 対応するマスクがクロップされた場合、一部のボックスが全てゼロになる可能性がある。
    # ここでそれらをフィルタリングする
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # バウンディングボックス。対応するマスクがクロップされた場合、
    # 一部のボックスが全てゼロになる可能性がある。
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # アクティブクラス
    # 異なるデータセットは異なるクラスを持つため、
    # この画像のデータセットでサポートされるクラスを追跡する。
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # メモリ使用量を減らすためマスクを小さなサイズにリサイズ
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # 画像メタデータ
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Stage 2分類器およびマスクヘッドの訓練用ターゲットを生成。
    これは通常の訓練では使用されない。デバッグやRPNヘッドを使わずに
    Mask RCNNヘッドを訓練するのに有用。

    入力:
    rpn_rois: [N, (y1, x1, y2, x2)] 提案ボックス。
    gt_class_ids: [instance count] 整数型クラスID
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] 正解マスク。フルサイズまたは
              ミニマスク可能。

    返し値:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]。整数型クラスID。
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]。クラス固有の
            bbox精密化。
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES)。bbox境界にクロップされ、
           ニューラルネットワーク出力サイズにリサイズされたクラス固有マスク。
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "intを期待したが{}を取得".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "intを期待したが{}を取得".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "boolを期待したが{}を取得".format(
        gt_masks.dtype)

    # GTボックスをROIに追加するのが一般的だが、ここでは行わない。
    # XinLei Chenの論文によると効果がないため。

    # gt_boxesとgt_masks部分の空のパディングをトリム
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "画像にはインスタンスが含まれている必要があります。"
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # ROIと正解ボックスの面積を計算。
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # 重なりを計算 [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # ROIをGTボックスに割り当て
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # 各ROIに割り当てられたGTボックス
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # ポジティブROIはGTボックスとのIoU >= 0.5のもの。
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # ネガティブROIは最大IoU 0.1-0.5のもの（ハード例マイニング）
    # TODO: ハード例マイニングするかしないか、それが問題
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # ROIをサブサンプル。前景33%を目指す。
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # 保持するROIのインデックスを結合
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # もっと必要？
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # 望ましい数を維持するのに十分なサンプルがないようだ
        # バランス。要求を緩和し残りを埋める。これは
        # Mask RCNN論文とは異なる可能性がある。

        # fgもbgもサンプルがない可能性がわずかにある。
        if keep.shape[0] == 0:
            # より簡単なIoU閾値でbg領域を選択
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # 残りを繰り返し背景roiで埋める。
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # BG ROIに割り当てられたgtボックスをリセット。
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # 保持された各ROIにclass_idを割り当て、FG ROIにはbbox refinementも追加。
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # クラス認識bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # bbox精密化を正規化
    bboxes /= config.BBOX_STD_DEV

    # クラス固有ターゲットマスクを生成
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # 画像サイズのマスクプレースホルダーを作成
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GTボックス
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # ミニマスクをGTボックスサイズにリサイズ
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # ミニバッチをプレースホルダーに配置
            class_mask = placeholder

        # マスクの一部を取得してリサイズ
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """アンカーとGTボックスが与えられた場合、重なりを計算し、ポジティブ
    アンカーとそれらを対応するGTボックスに一致するように精密化するデルタを識別する。

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] 整数型クラスID。
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) アンカーとGTボックスの一致。
               1 = ポジティブアンカー、-1 = ネガティブアンカー、0 = ニュートラル
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] アンカーbboxデルタ。
    """
    # RPNマッチ: 1 = ポジティブアンカー、-1 = ネガティブアンカー、0 = ニュートラル
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPNバウンディングボックス: [画像あたりの最大アンカー数, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # COCO群衆を処理
    # COCOの群衆ボックスは複数のインスタンスを囲むバウンディングボックス。
    # 訓練から除外する。群衆ボックスには負のクラスIDが与えられる。
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # 正解クラスIDとボックスから群衆をフィルタリング
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # 群衆ボックスとの重なりを計算 [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # 全てのアンカーが群衆と交差しない
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # 重なりを計算 [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # アンカーをGTボックスにマッチ
    # アンカーがGTボックスとのIoU >= 0.7で重なる場合、ポジティブ。
    # アンカーがGTボックスとのIoU < 0.3で重なる場合、ネガティブ。
    # ニュートラルアンカーは上記条件に一致しないもので、
    # 損失関数に影響を与えない。
    # ただし、GTボックスを未マッチのままにしない（稀だが発生）。代わりに、
    # 最も近いアンカーにマッチさせる（最大IoUが< 0.3でも）。
    #
    # 1. 最初にネガティブアンカーを設定。GTボックスが
    # マッチした場合は以下で上書きされる。群衆エリアのボックスをスキップ。
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. 各GTボックスにアンカーを設定（IoU値に関係なく）。
    # 複数のアンカーが同じIoUを持つ場合、すべてをマッチ
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. 高い重複を持つアンカーをポジティブに設定。
    rpn_match[anchor_iou_max >= 0.7] = 1

    # ポジティブとネガティブアンカーのバランスを取るためサブサンプル
    # ポジティブがアンカーの半分を超えないようにする
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # 余ったものを中立にリセット
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # ネガティブ提案も同様
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # 余ったものを中立にリセット
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # ポジティブアンカーについて、対応するGTボックスにマッチさせるために
    # 必要なシフトとスケールを計算。
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: ここでコードを重複するよりもbox_refinement()を使用
    for i, a in zip(ids, anchors[ids]):
        # 最も近いgtボックス（IoU < 0.7の可能性あり）
        gt = gt_boxes[anchor_iou_argmax[i]]

        # 座標を中心点＋幅/高さに変換。
        # GTボックス
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # アンカー
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # RPNが予測すべきバウンディングボックスの精密化を計算。
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # 正規化
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """領域提案ネットワークが生成するようなROI提案を生成する。

    image_shape: [Height, Width, Depth]
    count: 生成するROIの数
    gt_class_ids: [N] 整数型の正解クラスID
    gt_boxes: [N, (y1, x1, y2, x2)] ピクセル単位の正解ボックス。

    Returns: [count, (y1, x1, y2, x2)] ピクセル単位のROIボックス。
    """
    # プレースホルダー
    rois = np.zeros((count, 4), dtype=np.int32)

    # GTボックス周辺にランダムなROIを生成（countの90%）
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # ランダムな境界
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # 面積ゼロのボックスの生成を防ぐため、必要な数の2倍を生成して
        # 余分をフィルタリング。必要な数よりも有効なボックスが少ない場合、
        # ループして再試行。
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # ゼロ面積ボックスをフィルタリング
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # 軸1でソートしてx1 <= x2おy1 <= y2を保証し、その後
        # x1, y1, x2, y2の順序に再整形
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # 画像内のどこにでもランダムなROIを生成（カウントの10%）
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # ゼロ面積ボックスの生成を避けるため、必要な分の2倍を生成し
    # 余分をフィルタリング。必要な分より少ない有効ボックスしか得られない場合は
    # ループして再試行。
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # ゼロ面積ボックスをフィルタリング
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # 軸1でソートしてx1 <= x2おy1 <= y2を保証し、その後
    # x1, y1, x2, y2の順序に再整形
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """画像と対応するターゲットクラスID、
    バウンディングボックスデルタ、およびマスクを返すジェネレーター。

    dataset: データを取得するためのDatasetオブジェクト
    config: モデル設定オブジェクト
    shuffle: Trueの場合、各エポック前にサンプルをシャッフルする
    augment: (非推奨。代わりにaugmentationを使用)。trueの場合、ランダムな
        画像拡張を適用。現在は水平反転のみ提供。
    augmentation: オプション。imgaug (https://github.com/aleju/imgaug) 拡張。
        例えば、imgaug.augmenters.Fliplr(0.5)を渡すことで50%の確王で画像を
        左右反転させる。
    random_rois: 0より大きい場合、ネットワーク分類器とマスクヘッドの訓練に
                 使用する提案を生成。RPNなしでMask RCNN部分を訓練する場合に有用。
    batch_size: 各呼び出しで返す画像数
    detection_targets: Trueの場合、検出ターゲット(クラスID、bbox
        デルタ、マスク)を生成。通常はデバッグや可視化用で、訓練時は
        検出ターゲットはDetectionTargetLayerで生成される。
    no_augmentation_sources: オプション。拡張から除外するソースのリスト。
        ソースはデータセットを識別する文字列で、Datasetクラスで定義される。

    Pythonジェネレーターを返す。next()を呼び出すと、
    ジェネレーターはinputsとoutputsの2つのリストを返す。リストの内容は
    受け取った引数によって異なる:
    inputsリスト:
    - images: [batch, H, W, C]
    - image_meta: [batch, (メタデータ)] 画像詳細。compose_image_meta()を参照
    - rpn_match: [batch, N] 整数 (1=ポジティブアンカー, -1=ネガティブ, 0=ニュートラル)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] アンカーbboxデルタ。
    - gt_class_ids: [batch, MAX_GT_INSTANCES] 整数型クラスID
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]。高さと幅は
                use_mini_maskがTrueでない場合は画像のもので、Trueの場合は
                MINI_MASK_SHAPEで定義される。

    outputsリスト: 通常の訓練では通常空。しかしdetection_targetsが
        Trueの場合、outputsリストにはターゲットclass_ids、bboxデルタ、
        マスクが含まれる。
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # アンカー
    # [anchor_count, (y1, x1, y2, x2)]
    # ステップ1: 特徴マップサイズ計算
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Kerasはジェネレーターが無期限に実行されることを要求します。
    while True:
        try:
            # 次の画像を選ぶためインデックスを増分。エポックの開始時にシャッフル。
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # 画像のGTバウンディングボックスとマスクを取得。
            image_id = image_ids[image_index]

            # 画像ソースが拡張されない場合、拡張としてNoneを渡す
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=config.USE_MINI_MASK)

            # インスタンスを持たない画像をスキップ。これはクラスのサブセットで
            # 訓練し、画像に気にするクラスが含まれていない場合に発生する。
            if not np.any(gt_class_ids > 0):
                continue

            # RPNターゲット
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNNターゲット
            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # バッチ配列を初期化
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # 配列に入りきらないインスタンスがある場合、それらからサブサンプル。
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # バッチに追加
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # バッチは満杯か？
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Kerasはoutputとtargetsが同じ次元数を持つことを要求する
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # ログを出力して画像をスキップ
            logging.exception("画像処理エラー {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNNクラス
############################################################

class MaskRCNN():
    """Mask RCNNモデルの機能をカプセル化する。

    実際Kerasモデルはkeras_modelプロパティにある。
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: "training"または"inference"
        config: Configクラスのサブクラス
        model_dir: 訓練ログと訓練済み重みを保存するディレクトリ
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Mask R-CNNアーキテクチャを構築する。
            input_shape: 入力画像の形状。
            mode: "training"または"inference"。モデルの入力と
                出力はそれに応じて異なる。
        """
        assert mode in ['training', 'inference']

        # 画像サイズは2で何度も割り切れる必要がある
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("画像サイズは少なくとも6回、2で割り切れる必要があります。"
                            "ダウンスケーリングとアップスケーリング時の端数を避けるためです。"
                            "例: 256, 320, 384, 448, 512等を使用してください。")

        # 入力
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            # Region Proposal Network、画像のどこに物体がありそうかを大まかに特定
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            # バウンディングボックス回帰ターゲット
            # アンカーを「どのくらい」「どの方向に」動かせば正解になるかを学習するためのターゲット値
            # アンカーは事前に定義された固定サイズのボックス
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # 検出 GT (クラスID、バウンディングボックス、マスク)
            # 1. GTクラスID (ゼロパディング)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. ピクセル単位のGTボックス (ゼロパディング)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] 画像座標系
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # 座標を正規化
            # 1. gt_boxes
            gt_boxes = KL.Lambda(
                # ボックスを正規化して数値安定化と学習を効率化
                lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3]),
                output_shape=lambda s: s
            )(input_gt_boxes)
            # 3. GTマスク (ゼロパディング)
            # インスタンスセグメンテーション用の正解マスクデータ
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # 正規化座標のアンカー
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # 共有畳み込みレイヤーを構築。
        # ボトムアップレイヤー
        # 各ステージの最後のレイヤーのリストを返す、合計5つ。
        # ヘッド(stage 5)を作成しないため、リストの4番目の項目を選択。
        # C2〜C5はresnetのグラフを5分割したネットワーク
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)
        # トップダウンレイヤー
        # C5（最も深い特徴マップ）を1x1畳み込みで256チャネルに変換し、P5を作成
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        # P4を作成：P5を2倍にアップサンプリングした特徴とC4を1x1畳み込みした特徴を加算
        P4 = KL.Add(name="fpn_p4add")([
            # P5を2倍にアップサンプリング（解像度を上げる）してP4のサイズに合わせる
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            # C4を1x1畳み込みで256チャネルに変換
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        # P3を作成：P4を2倍にアップサンプリングした特徴とC3を1x1畳み込みした特徴を加算
        P3 = KL.Add(name="fpn_p3add")([
            # P4を2倍にアップサンプリング（解像度を上げる）してP3のサイズに合わせる
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            # C3を1x1畳み込みで256チャネルに変換
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        # P2を作成：P3を2倍にアップサンプリングした特徴とC2を1x1畳み込みした特徴を加算
        P2 = KL.Add(name="fpn_p2add")([
            # P3を2倍にアップサンプリング（解像度を上げる）してP2のサイズに合わせる
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            # C2を1x1畳み込みで256チャネルに変換
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # 最終特徴マップを得るため、全てのPレイヤーに3x3 convをアタッチ。
        # P2に3x3畳み込みを適用してエイリアシング効果を軽減し、最終的な特徴マップを生成
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        # P3に3x3畳み込みを適用してエイリアシング効果を軽減し、最終的な特徴マップを生成
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        # P4に3x3畳み込みを適用してエイリアシング効果を軽減し、最終的な特徴マップを生成
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        # P5に3x3畳み込みを適用してエイリアシング効果を軽減し、最終的な特徴マップを生成
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6はRPNの5番目のアンカースケールに使用。stride 2で
        # P5からサブサンプリングして生成。
        # P6を作成：P5を最大プーリング（ストライド2）でダウンサンプリングして、より大きな物体検出用
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # P6はRPNで使用されるが、分類器ヘッドでは使用されないことに注意。
        # RPN（Region Proposal Network）で使用する特徴マップのリスト（P2〜P6の5スケール）
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        # Mask R-CNN（分類とマスク生成）で使用する特徴マップのリスト（P2〜P5の4スケール、P6は低解像度なので除外）
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Kerasが必要とするため、バッチ次元で複製
            # TODO: アンカーの複製を避けるように最適化できるか？
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # Kerasの定数に対する悪いサポートを回避するハック
            anchors = KL.Lambda(
                lambda x: tf.convert_to_tensor(anchors, dtype=tf.float32),  # tf.Variable → convert_to_tensor で安全に定数化
                output_shape=lambda s: anchors.shape,
                name="anchors"
            )(input_image)
        else:
            anchors = input_anchors

        # RPNモデル
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # 各特徴マップに対してRPNモデルを適用
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # logits1, plobs1, bbox1, logits2, plob2,bbox2 → logits1,2, plobs1,2,
        # 例: [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # 提案を生成
        # 提案は[batch, N, (y1, x1, y2, x2)]の正規化座標で
        # ゼロパディングされている。
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        # ProposalLayerのcallメソッドが実行され、NMSを実行する
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # 画像が由来するデータセットでサポートされるクラスIDをマークする
            # クラスIDマスク。
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"],
                output_shape=lambda s: (s[0], None))(input_image_meta)
            # デバッグなどの用途
            if not config.USE_RPN_ROIS:
                # RPNとNMSで予測されたROIを無視し、入力として提供されたROIを使用。
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # 座標を正規化
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]),
                    output_shape=lambda s: s)(input_rois)
            else:
                target_rois = rpn_rois

            # 検出ターゲットを生成
            # 提案をサブサンプルし、訓練用のターゲット出力を生成
            # 提案クラスID、gt_boxes、gt_masksはゼロパディングされていることに注意。
            # 同様に、返されるroisとtargetsもゼロパディングされている。
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # ネットワークヘッド
            # TODO: これがゼロパディングされたROIを適切に処理するか確認
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            # TODO: クリーンアップ (必要に応じてtf.identifyを使用)
            output_rois = KL.Lambda(lambda x: x * 1,
                                   output_shape=lambda s: s,
                                   name="output_rois")(rois)

            # 損失es
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x),
                                      output_shape=lambda s: (),
                                      name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x),
                                     output_shape=lambda s: (),
                                     name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x),
                                  output_shape=lambda s: (),
                                  name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x),
                                 output_shape=lambda s: (),
                                 name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x),
                                 output_shape=lambda s: (),
                                 name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # モデル
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # ネットワークヘッド
            # 提案分類器およびBBox回帰ヘッド
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # 検出
            # 出力は[batch, num_detections, (y1, x1, y2, x2, class_id, score)]で
            # 正規化座標
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # 検出結果用のマスクを作成
            detection_boxes = KL.Lambda(lambda x: x[..., :4],
                                       output_shape=lambda s: (s[0], s[1], 4))(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # マルチGPUサポートを追加。
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """モデルディレクトリ内の最後に訓練されたモデルの最後のチェックポイントファイルを検索。
        Returns:
            最後のチェックポイントファイルのパス
        """
        # ディレクトリ名を取得。各ディレクトリはモデルに対応
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "{}でモデルディレクトリが見つかりませんでした".format(self.model_dir))
        # 最後のディレクトリを選択
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # 最後のチェックポイントを検索
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "{}で重みファイルが見つかりませんでした".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """対応するKeras関数の改変版。
        マルチGPUサポートと一部のレイヤーを読み込みから除外する機能を追加。
        exclude: 除外するレイヤー名のリスト
        """
        import h5py
        # TensorFlow 2.x / Keras統合用のインポート
        try:
            from tensorflow.python.keras.saving import hdf5_format as saving
        except ImportError:
            try:
                from keras.engine import saving
            except ImportError:
                # Keras 2.2以前は'topology'名前空間を使用した。
                from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # マルチGPU訓練では、モデルをラップする。重みを持つ
        # 内部モデルのレイヤーを取得。
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # 一部のレイヤーを除外
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # ログディレクトリを更新
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """KerasからImageNet訓練済み重みをダウンロード。
        重みファイルのパスを返す。
        """
        try:
            from tensorflow.keras.utils import get_file
        except ImportError:
            from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """モデルを訓練のために準備。損失、正則化、メトリクスを追加。
        その後、Kerasのcompile()関数を呼び出す。
        """
        # オプティマイザーオブジェクト
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # 損失を追加
        # まず、重複を避けるために以前に設定された損失をクリア
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # L2正則化を追加
        # バッチ正規化レイヤーのgammaとbeta重みをスキップ。
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # コンパイル
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # 損失用のメトリクスを追加
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """正規表現にマッチする名前のモデルレイヤーを訓練可能に設定する。
        """
        # 最初の呼び出しでメッセージを出力（再帰呼び出しではない）
        if verbose > 0 and keras_model is None:
            log("訓練するレイヤーを選択中")

        keras_model = keras_model or self.keras_model

        # マルチGPU訓練では、モデルをラップする。重みを持つ
        # 内部モデルのレイヤーを取得。
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # レイヤーはモデルか？
            if layer.__class__.__name__ == 'Model':
                print("モデル内: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # 訓練可能か？
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # レイヤーを更新。レイヤーがコンテナの場合、内部レイヤーを更新。
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # 訓練可能レイヤー名を出力
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """モデルのログディレクトリとエポックカウンターを設定する。

        model_path: Noneまたはこのコードが使用する形式と異なる場合、
            新しいログディレクトリを設定しエポックを0から開始。そうでない場合、
            ファイル名からログディレクトリとエポックカウンターを抽出。
        """
        # 新しいモデルを開始するかのように日付とエポックカウンターを設定
        self.epoch = 0
        now = datetime.datetime.now()

        # 日付とエポックを含むモデルパスがある場合はそれらを使用
        if model_path:
            # 中断したところから続行。ファイル名からエポックと日付を取得
            # モデルパスの例:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # ファイル内のエポック番号は1ベースで、Kerasコードでは0ベース。
                # そのため調整し、次のエポックから開始するために1を加算
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # 訓練ログ用ディレクトリ
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # 各エポック後に保存するパス。Kerasによって埋められるプレースホルダーを含む。
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """モデルを訓練する。
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "モデルを訓練モードで作成してください。"

        # 事前定義されたレイヤー正規表現
        layer_regex = {
            # バックボーン以外のすべてのレイヤー
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # 特定のResnetステージ以上
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # すべてのレイヤー
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # データジェネレーター
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # 存在しない場合はlog_dirを作成
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # コールバック
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # カスタムコールバックをリストに追加
        if custom_callbacks:
            callbacks += custom_callbacks

        # 訓練
        log("\nエポック{}で開始。学習率={}\n".format(self.epoch, learning_rate))
        log("チェックポイントパス: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Windows対応: Kerasはマルチプロセッシングワーカーを使用するとWindowsで失敗。
        # 詳細はここを参照:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """画像のリストを取り、ニューラルネットワークへの入力として
        期待される形式に変更する。
        images: 画像行列[height,width,depth]のリスト。画像のサイズは
            異なっていてもよい。

        3つのNumpy行列を返す:
        molded_images: [N, h, w, 3]。リサイズおよび正規化された画像。
        image_metas: [N, メタデータの長さ]。各画像の詳細情報。
        windows: [N, (y1, x1, y2, x2)]。元の画像を持つ画像の部分
            （パディングを除く）。
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # 画像をリサイズ
            # TODO: リサイズをmold_image()に移動
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # image_metaを構築
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # 追加
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # 配列にパック
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """一枚の画像の検出結果をニューラルネットワーク出力の形式から
        アプリケーションの他の部分で使用するのに適した形式に再フォーマットする。

        detections: [N, (y1, x1, y2, x2, class_id, score)] 正規化座標
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] リサイズ前の元の画像形状
        image_shape: [H, W, C] リサイズおよびパディング後の画像形状
        window: [y1, x1, y2, x2] パディングを除いた実際の画像がある
                画像内のボックスのピクセル座標。

        Returns:
        boxes: [N, (y1, x1, y2, x2)] ピクセル単位のバウンディングボックス
        class_ids: [N] 各バウンディングボックスの整数型クラスID
        scores: [N] class_idの浮動小数点確率スコア
        masks: [height, width, num_instances] インスタンスマスク
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """検出パイプラインを実行する。

        images: サイズが異なる可能性のある画像のリスト。

        画像ごとに1つずつ辞書を含む辞書のリストを返す。辞書の内容:
        rois: [N, (y1, x1, y2, x2)] 検出バウンディングボックス
        class_ids: [N] 整数型クラスID
        scores: [N] クラスIDの浮動小数点確率スコア
        masks: [H, W, N] インスタンスバイナリマスク
        """
        assert self.mode == "inference", "モデルを推論モードで作成してください。"
        assert len(
            images) == self.config.BATCH_SIZE, "len(images)はBATCH_SIZEと等しくなければなりません"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "リサイズ後、すべての画像は同じサイズである必要があります。IMAGE_RESIZE_MODEと画像サイズを確認してください。"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """検出パイプラインを実行するが、すでに成型された入力を期待する。
        主にデバッグとモデルの検査に使用。

        molded_images: load_image_gt()を使用して読み込んだ画像のリスト
        image_metas: 画像メタデータ、load_image_gt()からも返される

        画像ごとに1つずつ辞書を含む辞書のリストを返す。辞書の内容:
        rois: [N, (y1, x1, y2, x2)] 検出バウンディングボックス
        class_ids: [N] 整数型クラスID
        scores: [N] クラスIDの浮動小数点確率スコア
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "モデルを推論モードで作成してください。"
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "画像数はBATCH_SIZEと等しくなければなりません"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "画像は同じサイズである必要があります"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """指定された画像サイズのアンカーピラミッドを返す。"""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # アンカーをキャッシュし、画像の形状が同じ場合は再使用
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # アンカーを生成
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # inspect_modelノートブックで使用されるため、最新のアンカーの
            # ピクセル座標コピーを保持。
            # TODO: ノートブックがこれを使用しないようにリファクタリングされた後に除去
            self.anchors = a
            # 座標を正規化
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """計算グラフ内のTFテンソルの祖先を検索。
        tensor: TensorFlowシンボリックテンソル。
        name: 検索する祖先テンソルの名前
        checked: 内部使用。グラフを辿る際のループを避けるため、
                 すでに検索されたテンソルのリスト。
        """
        checked = checked if checked is not None else []
        # 非常に長いループを避けるため、深さに制限を設ける
        if len(checked) > 500:
            return None
        # Kerasが自動的に追加するため、名前をregexに変換し、数字プレフィックスの
        # マッチングを許可
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """レイヤーが別のレイヤーにカプセル化されている場合、この関数は
        カプセル化を揘り下げ、重みを保持するレイヤーを返す。
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """重みを持つレイヤーのリストを返す。"""
        layers = []
        # すべてのレイヤーをループ
        for l in self.keras_model.layers:
            # レイヤーがラッパーの場合、内部の訓練可能レイヤーを検索
            l = self.find_trainable_layer(l)
            # 重みを持つ場合はレイヤーを含める
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """与えられた出力を計算する計算グラフのサブセットを実行。

        image_metas: 提供された場合、画像はすでに成型されている（つまりリサイズ、
            パディング、正規化済み）と仮定

        outputs: 計算する(name, tensor)タプルのリスト。テンソルは
            シンボリックTensorFlowテンソルで、名前は簡単な追跡用。

        結果の順序付き辞書を返す。キーは入力で受け取った名前で、値はNumpy配列。
        """
        model = self.keras_model

        # 希望する出力を順序付き辞書に整理
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # 計算グラフの一部を実行するKeras関数を構築
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # 入力を準備
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # 推論を実行
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # 生成されたNumpy配列を辞書にパックし、結果をログ出力。
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  データフォーマット
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """画像の属性を受け取り、1次元配列に格納。

    image_id: 画像のint ID。デバッグに有用。
    original_image_shape: リサイズやパディング前の[H, W, C]。
    image_shape: リサイズおよびパディング後の[H, W, C]
    window: ピクセル単位の(y1, x1, y2, x2)。実際の画像がある画像の領域
            (パディングを除く)
    scale: 元の画像に適用されたスケールファクター (float32)
    active_class_ids: 画像が由来するデータセットで利用可能なclass_idsのリスト。
        すべてのクラスがすべてのデータセットに存在しない複数のデータセットの
        画像で訓練する場合に有用。
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """画像属性を含む配列をそのコンポーネントに解析。
    詳細はcompose_image_meta()を参照。

    meta: [batch, meta length] meta lengthはNUM_CLASSESに依存

    解析された値の辞書を返す。
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """画像属性を含むテンソルをそのコンポーネントに解析。
    詳細はcompose_image_meta()を参照。

    meta: [batch, meta length] meta lengthはNUM_CLASSESに依存

    解析されたテンソルの辞書を返す。
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """RGB画像（または画像の配列）を期待し、
    平均ピクセルを減算してfloatに変換。画像の色は
    RGB順であることを期待。
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """mold()で正規化された画像を受け取り、元の画像を返す。"""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  その他のグラフ関数
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """ボックスはしばしば[N, 4]の形状の行列で表現され、
    ゼロでパディングされる。これはゼロボックスを除去。

    boxes: [N, 4] ボックスの行列。
    non_zeros: [N] 保持する行を識別する1Dブールマスク
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """countsの値に応じて、xの各行から
    異なる数の値を選択。
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """ボックスをピクセル座標から正規化座標に変換。
    boxes: [..., (y1, x1, y2, x2)] ピクセル座標
    shape: [..., (height, width)] ピクセル単位

    注意: ピクセル座標では(y2, x2)はボックスの外側。しかし正規化座標では
    ボックスの内側。

    Returns:
        [..., (y1, x1, y2, x2)] 正規化座標
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """ボックスを正規化座標からピクセル座標に変換。
    boxes: [..., (y1, x1, y2, x2)] 正規化座標
    shape: [..., (height, width)] ピクセル単位

    注意: ピクセル座標では(y2, x2)はボックスの外側。しかし正規化座標では
    ボックスの内側。

    Returns:
        [..., (y1, x1, y2, x2)] ピクセル座標
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
