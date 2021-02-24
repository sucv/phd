import cv2
import numpy as np
import os
import pickle
import time
import h5py
from PIL import Image
from tqdm import tqdm
import scipy.io as sio
from memory_profiler import profile


def creating_data(num_image=10000, dim=120, channel=3):

    # Create 10000 static images
    for i in tqdm(range(num_image)):
        image = np.random.randint(0, 254, (dim, dim, channel), dtype=np.uint8)
        image = Image.fromarray(image)
        image.save("data/static/" + str(i) + ".jpg")
    print("Data in static jpg format has been created!")

    # Create a video with 10000 frames
    # In mp4 format
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(filename="data/mp4.mp4", fourcc=codec, fps=64, frameSize=(dim, dim), isColor=True)
    for i in tqdm(range(num_image)):
        frame = np.random.randint(0, 254, (dim, dim, channel), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    print("Data in mp4 format has been created!")



    # In plain hdf5/h5 format
    video = np.random.randint(0, 254, (num_image, channel, dim, dim), dtype=np.uint8)
    with h5py.File("data/h5_plain.h5", "w") as data_file:
        data_file.create_dataset(name="video", data=video, compression="gzip", compression_opts=9)
    print("Data in plain hdf5/h5 format has been created!")

    # In chunked hdf5/h5 format
    hdf5_file = h5py.File('data/h5_chunked.h5', 'w')

    first_frame = np.random.randint(0, 254, (channel, dim, dim), dtype=np.uint8)
    hdf5_dataset = hdf5_file.create_dataset('video', data=first_frame[None, ...], maxshape=(
        None, first_frame.shape[0], first_frame.shape[1], first_frame.shape[2]), chunks=True, compression="gzip", compression_opts=9)

    for i in tqdm(range(num_image-1)):
        image = np.random.randint(0, 254, (channel, dim, dim), dtype=np.uint8)
        hdf5_dataset.resize(hdf5_dataset.len() + 1, axis=0)
        hdf5_dataset[hdf5_dataset.len() - 1] = image
    hdf5_file.close()
    print("Data in chunked hdf5/h5 format has been created!")

    # In mat format
    sio.savemat("data/mat.mat", {'video': video})
    print("Data in mat format has been created!")

    # In binary format
    with open("data/binary.bin", "wb") as file:
        bytes(video)
        file.write(bytes(video))
    print("Data in binary format has been created!")

    # In pickle format
    with open('data/pickle.pkl', 'wb') as handle:
        pickle.dump(video, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Data in plain pickle format has been created!")

    # In npy format
    np.save('data/npy.npy', video)
    print("Data in npy format has been created!")


@profile
def load_static():
    data_matrix = np.zeros((2500, 3, 120, 120), dtype=np.uint8)
    for i, file in enumerate(os.listdir("data/static")):
        if i % 4 == 0:
            filename = os.path.join("data/static", file)
            data_matrix[i //4] = np.asarray(Image.open(filename)).transpose((2, 0, 1))

@profile
def load_mp4():
    data_matrix = np.zeros((2500, 3, 120, 120), dtype=np.uint8)
    cap = cv2.VideoCapture('data/mp4.mp4')
    for i in range(10000):
        ret, frame = cap.read()
        if i % 4 == 0:
            data_matrix[i // 4] = frame.transpose((2, 0, 1))


@profile
def load_plain_h5():
    with h5py.File("data/h5_plain.h5", "r") as f:
        data_matrix = f['video'][()][::4]

@profile
def load_chunked_h5():
    with h5py.File("data/h5_chunked.h5", "r") as f:
        data_matrix = f['video'][()][::4]

@profile
def load_pickle():
    with open('data/pickle.pkl', 'rb') as f:
        data_matrix = pickle.load(f)[::4]

@profile
def load_mat():
    mat_content = sio.loadmat("data/mat.mat")
    data_matrix = mat_content['video'][::4]

@profile
def load_binary():
    data_matrix = np.fromfile("data/binary.bin")[::4]

@profile
def load_npy():
    data_matrix = np.load("data/npy.npy")[::4]

@profile
def load_npy_with_memory_map_mode():
    data_matrix = np.load("data/npy.npy", mmap_mode="c")[::4]


if __name__ == '__main__':
    # creating_data(num_image=10000, dim=120, channel=3)

    start = time.time()
    load_static()
    elapsed = time.time() - start
    print("Elapsed for loading static: {:4f}s\n".format(elapsed))

    start = time.time()
    load_mp4()
    elapsed = time.time() - start
    print("Elapsed for loading mp4: {:4f}s\n".format(elapsed))

    start = time.time()
    load_binary()
    elapsed = time.time() - start
    print("Elapsed for loading binary: {:4f}s\n".format(elapsed))

    start = time.time()
    load_mat()
    elapsed = time.time() - start
    print("Elapsed for loading mat: {:4f}s\n".format(elapsed))

    start = time.time()
    load_pickle()
    elapsed = time.time() - start
    print("Elapsed for loading pickle: {:4f}s\n".format(elapsed))

    start = time.time()
    load_plain_h5()
    elapsed = time.time() - start
    print("Elapsed for loading plain hdf5/h5: {:4f}s\n".format(elapsed))

    start = time.time()
    load_chunked_h5()
    elapsed = time.time() - start
    print("Elapsed for loading chunked hdf5/h5: {:4f}s\n".format(elapsed))

    start = time.time()
    load_npy()
    elapsed = time.time() - start
    print("Elapsed for loading npy: {:4f}s\n".format(elapsed))

    start = time.time()
    load_npy_with_memory_map_mode()
    elapsed = time.time() - start
    print("Elapsed for loading npy with mempry map mode: {:4f}s\n".format(elapsed))
