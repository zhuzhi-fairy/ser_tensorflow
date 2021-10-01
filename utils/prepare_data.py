import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

def load_audio_tf(filepath, ch, sr, **kwargs):
    '''load the audio tensor \
    Args: \
        filepath: path of .wav file \
        ch: target channel of input signal \
            mean signal of all channel when ch=-1 \
        sr: target sampling rate \
    Return: \
        audio_tensor: tf.tensor of the audio signal \
    '''
    # load audio, output shape [len, ch]
    audioIOtensor = tfio.audio.AudioIOTensor(filepath, dtype=tf.int16)
    # to float32
    audio_tensor = tf.cast(audioIOtensor.to_tensor(), tf.float32)
    # channels
    if ch == -1:
        audio_tensor = tf.math.reduce_mean(audio_tensor, axis=-1)
    else:
        audio_tensor = audio_tensor[:, ch]
    # normalization
    m, s = tf.math.reduce_mean(audio_tensor), tf.math.reduce_std(audio_tensor)
    audio_tensor = (audio_tensor-m)/s
    # resample
    if audioIOtensor.rate != sr:
        audio_tensor = tfio.audio.resample(
            input=audio_tensor,
            rate_in=tf.cast(audioIOtensor.rate, tf.int64),
            rate_out=tf.constant(sr, tf.int64)
        )
    return audio_tensor


def segment_tf(audio_tensor, seg_len_p, seg_hop_p, **kwargs):
    '''Segmenting audio signals \
    Args: \
        audio_tensor: input tf.Tensor \
        seg_len_p: segment length in sample points \
        seg_hop_p: segment hop in sample points \
    Return: \
        segments: tf.Tensor of segments 
            shape: (number of segments, segment length)
    '''
    segments_list, audio_length = [], int(audio_tensor.shape[0])
    # cropping and padding signal
    start, end = 0, seg_len_p
    while(end < audio_length):
        # cropping
        segments_list.append(audio_tensor[start:end])
        start += seg_hop_p
        end += seg_hop_p
    # zero-padding the last segment
    audio_tensor_end = tf.pad(audio_tensor[start:end], [[0, end-audio_length]])
    segments_list.append(audio_tensor_end)
    segments = tf.stack(segments_list)
    return segments


def logmelspectrogram_tf(audio_tensor, sr, win_len_p, win_hop_p, n_mels, **kwargs):
    '''Calculate log-mel-spectrogram \
    Args: \
        audio_tensor: input tf.Tensor \
        sr: sampling rate \
        win_len_p: window length in sample points \
        win_hop_p: window hop in sample points \
        n_mels: number of mel-filters
    Return: \
        db_mel_spectrogram: (n_mels, n_frames)
    '''
    if tfio.__version__ == '0.20.0':
        tfio_audio = tfio.audio
    else:
        tfio_audio = tfio.experimental.audio
    # spectrogram
    nfft = int(2**np.ceil(np.log(win_len_p) / np.log(2.0)))
    spectrogram = tfio_audio.spectrogram(
        input=audio_tensor,
        nfft=nfft,
        window=win_len_p,
        stride=win_hop_p
    )
    # melspectrogram
    mel_spectrogram = tfio_audio.melscale(
        input=spectrogram,
        rate=sr,
        mels=n_mels,
        fmin=80,
        fmax=int(sr/2-200)
    )
    # dB scale
    db_mel_spectrogram = tfio_audio.dbscale(
        input=mel_spectrogram, top_db=90)
    # expand dim
    return tf.expand_dims(db_mel_spectrogram, -1)


def label_sample_weight(orig_labels, num_segs, sample_weight_mode, **kwargs):
    '''Make label and sample weight tensors \
    Args: \
        orig_labels: orginal label tensors 
            (number of files, number of classes) \
        num_segs: list of the number of segments \
        sample_weight_mode: \
            noweight, segment, class, segment_class \
    Returns: \
        label_tensor: tensor of labels (number of segments, number of classes) \
        sample_weight_tensor: tensor of sample weights (number of segments,)
    '''
    num_class = orig_labels.shape[1]
    # make label tensor
    labels_list = []
    for n in range(len(num_segs)):
        label_t = tf.concat([tf.expand_dims(orig_labels[n], 0)]*num_segs[n], 0)
        labels_list.append(label_t)
    label_tensor = tf.concat(labels_list, 0)
    # make sample weight tensor
    if sample_weight_mode.endswith('class'):
        class_weights = orig_labels.sum() / orig_labels.sum(0) / num_class
        np.place(class_weights, class_weights==np.inf, 0)
        class_weight_tensor = tf.math.reduce_sum(label_tensor*class_weights, 1)
    if sample_weight_mode.startswith('segment'):
        segment_weight_tensor = tf.convert_to_tensor(
            [1/n for n in num_segs for _ in range(n)])
    if sample_weight_mode == 'class':
        sample_weight_tensor = class_weight_tensor
    elif sample_weight_mode == 'segment':
        sample_weight_tensor = segment_weight_tensor
    elif sample_weight_mode == 'segment_class':
        sample_weight_tensor = class_weight_tensor * segment_weight_tensor
    else:
        sample_weight_tensor = tf.ones((sum(num_segs)))
    return label_tensor, sample_weight_tensor


def segment_audio_dataset(df,
                          ch=-1,
                          sr=16000,
                          seg_len=3,
                          seg_hop=1,
                          sample_weight_mode='segment_class',
                          **kwargs):
    '''Make audio segments dataset \
    Args: \
        df: input dataframe includes filepath and label
        ch: channel
        sr: sampling rate
        seg_len: segment length in seconds
        seg_hop: segment hop in seconds
        sample_weight_mode: mode of sample weight
            noweight, segment, class, segment_class
    Returns:
        ds: tf.data.Dataset includes (segments, labels, sample weights)
    '''
    # parameters
    seg_len_p = seg_len * sr
    seg_hop_p = seg_hop * sr
    # audio data and original labels
    df = df.sort_values('filepath')
    file_list = df.filepath.to_list()
    orig_labels = np.stack(df.label.to_list()).astype('float32')
    # load audio
    audio_list = [load_audio_tf(file, ch, sr) for file in file_list]
    # segmenting
    segments_list = [segment_tf(at, seg_len_p, seg_hop_p) for at in audio_list]
    segment_tensor = tf.concat(segments_list, 0)
    num_segs = [seg.shape[0] for seg in segments_list]
    # label
    label_tensor, sample_weight_tensor = label_sample_weight(
        orig_labels, num_segs, sample_weight_mode)
    # make datasets
    ds_audio = tf.data.Dataset.from_tensor_slices(segment_tensor)
    ds_label = tf.data.Dataset.from_tensor_slices(label_tensor)
    ds_sample_weight = tf.data.Dataset.from_tensor_slices(sample_weight_tensor)
    return tf.data.Dataset.zip((ds_audio, ds_label, ds_sample_weight)), num_segs


def feature_dataset(ds,
                    sr=16000,
                    win_len=0.025,
                    win_hop=0.01,
                    n_mels=40,
                    **kwargs):
    '''Make feature dataset \
    Args: \
        ds: tf.data.Dataset includes (segments, labels, sample weights)
        sr: sampling rate
        win_len: window length in seconds
        win_hop: window hop in seconds
        n_mels: number of mel-filter
    Returns:
        ds_feature: tf.data.Dataset includes (feature, labels, sample weights)
    '''
    win_len_p = int(win_len * sr)
    win_hop_p = int(win_hop * sr)
    ds_feature = ds.map(
        lambda x, l, s: (logmelspectrogram_tf(
            x, sr, win_len_p, win_hop_p, n_mels), l, s),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds_feature
