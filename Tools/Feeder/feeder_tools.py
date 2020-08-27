import random
import numpy as np


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M), dtype=np.float32)
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)


    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def add_noise(data_numpy):
    C, T, V, M = data_numpy.shape
    for frame in range(T):
        xyz = data_numpy[:, frame, :, 0] # (3, 25)
        if xyz[0][0] != 0:
            xyz += np.random.uniform(-0.2, 0.2, (C, V))
        data_numpy[:, frame, :, 0] = xyz

    return data_numpy


def repeat_padding(data_numpy):
    data_tmp = np.transpose(data_numpy, [3,1,2,0])     # [2,300,25,3]
    for i_p, person in enumerate(data_tmp):
        if person.sum()==0:
            continue
        if person[0].sum()==0:
            index = (person.sum(-1).sum(-1)!=0)
            tmp = person[index].copy()
            person*=0
            person[:len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum()==0:
                if person[i_f:].sum()==0:
                    rest = len(person)-i_f
                    num = int(np.ceil(rest/i_f))
                    pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                    data_tmp[i_p,i_f:] = pad
                    break
    data_numpy = np.transpose(data_tmp, [3,1,2,0])
    return data_numpy


def frame_normalization(data_numpy):
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    center_frame = int((begin + end) / 2)

    # center_node = 0
    center_node = 20
    for c in range(3):
        origin = data_numpy[c, center_frame, center_node, 0]
        '''
        for frame in range(begin, end):
            temp = data_numpy[c, frame, :, 0]  # (25)
            temp = temp - origin
            data_numpy[c, frame, :, 0] = temp
        '''
        data_numpy[c, begin:end, :, 0] = data_numpy[c, begin:end, :, 0] - origin

    return data_numpy


def frame_thinning(data_numpy, frame_size):
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    len_valid_frame = end - begin

    data_numpy = data_numpy[:, begin:end:, :, :]
    if len_valid_frame <= (frame_size + 50):
        data_numpy = random_choose(data_numpy, frame_size)
    elif len_valid_frame <= (frame_size + 150):
        data_numpy = thinning(data_numpy, thin=int((frame_size + 150) // frame_size))
        data_numpy = random_choose(data_numpy, frame_size)
    else:
        data_numpy = thinning(data_numpy, thin=int((frame_size + 250) // frame_size))
        data_numpy = random_choose(data_numpy, frame_size)

    return data_numpy


def thinning(data_numpy, thin):
    data_numpy = data_numpy.transpose(1, 0, 2, 3)
    data_numpy = data_numpy[::thin]
    data_numpy = data_numpy.transpose(1, 0, 2, 3)

    return data_numpy


def valid_choose(data_numpy, frame_size):
    C, T, V, M = data_numpy.shape
    if T < frame_size:
        data_numpy = auto_pading(data_numpy, frame_size, random_pad=True)

    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    len_valid_frame = end - begin

    if len_valid_frame <= frame_size:
        return data_numpy[:, :frame_size, :, :]
    else:
        begin = random.randint(0, len_valid_frame - frame_size)
        return data_numpy[:, begin:begin + frame_size, :, :]
