import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from skimage import color


class BasicOperations:
    # class for basic image operations
    def __init__(self):
        pass

    def read_image(self, img_path):
        try:
            img = np.array(image.imread(img_path)).astype('float64')
            return img
        except:
            os.remove(img_path)
            print("file:", img_path, "removed")
            return False

    def normalize_img(self, np_img): # range: [0 -> 1]
        # the formula is: (x - min(x)) / (max(x) - min(x))
        if (np.max(np_img) == np.min(np_img)):
            return False
        return (np_img - np.min(np_img)) / (np.max(np_img) - np.min(np_img))

    def show_img(self, np_img, is_grayscale=False):
        np_img = self.normalize_img(np_img)
        plt.figure(figsize = (5,5))
        if is_grayscale:
            plt.imshow(np_img, cmap=plt.get_cmap('gray'))
        else:
            plt.imshow(np_img)
        plt.show()

    def convert_img_colors(self, dest_color, np_img):
        if dest_color == 'RGB':
            return color.gray2rgb(np_img)
        elif dest_color == 'GS':
            return color.rgb2gray(np_img)
        else:
            return False


class LoadData:
    # class for loading the data
    def __init__(self, dataset_path):
        self.bo_obj = BasicOperations()
        self.dataset_path = dataset_path

    def dataset_check(self):
        # check that all the images are in the same size
        img_shape = (0, 0)
        img_arr = []
        for filename in os.listdir(self.dataset_path):
            if (not filename.endswith(".txt")):
                img = self.bo_obj.read_image(os.path.join(self.dataset_path, filename))
                if (type(img) == bool): # reading-image function was crashed
                    continue
                img = self.bo_obj.convert_img_colors("GS", img)
                if (img_shape == (0, 0)):
                    img_shape = img.shape
                else:
                    if (img.shape != img_shape):
                        return False, []
                img_arr.append(img.flatten())
        print("read done")
        print("number of images: ", len(img_arr))
        return True, img_arr

    def build_faces_matrix(self):
        img_ds_name = "img_ds1.npy"
        img_params_name = "imgs_mean1.npy"
        if (os.path.isfile(img_ds_name)):
            print("img ds exists")
            return np.load(img_ds_name)
        else:
            is_dataset_ok, img_arr = self.dataset_check()
        if (is_dataset_ok):
            img_arr = np.asarray(img_arr).T
            mean_img = np.mean(img_arr, axis=1)
            mean_img = mean_img.reshape((mean_img.shape[0], 1))
            np.save(img_params_name, mean_img) # saving it for the recognition process
            img_arr = img_arr - mean_img # removing common features from all the images
            np.save(img_ds_name, img_arr)
            # self.bo_obj.show_img(mean_img.reshape((100,100)), True) # show mean img
            return img_arr
        return False


class PCA:
    # class for operating the PCA algorithm
    def __init__(self):
        pass

    def get_cov_matrix(self, img_arr):
        # calculating the cov-matrix and its eigenvectors-values
        cov_matrix = np.dot(img_arr.T, img_arr)
        return np.linalg.eig(cov_matrix)

    def get_K_biggest_eigenvectors(self, img_arr, eig_vectors, eig_values):
        # finding the K biggest eigenvectors (according to their corresponding eigenvalues)
        eig_vectors_sorted = np.take(eig_vectors, np.argsort(eig_values), axis=1)
        K_biggest = int(0.1*eig_vectors_sorted.shape[1])
        eig_vectors_biggest = eig_vectors_sorted[:, (eig_vectors_sorted.shape[1] - K_biggest):]
        eig_vectors_biggest = np.dot(img_arr, eig_vectors_biggest)
        return K_biggest, eig_vectors_biggest 

    def calc_pca_eigenfaces(self, img_arr):
        eig_values, eig_vectors = self.get_cov_matrix(img_arr)
        return self.get_K_biggest_eigenvectors(img_arr, eig_vectors, eig_values)


class RecognizeFaces:
    # class for the face recognition final result
    def __init__(self, K_biggest, eig_vectors_biggest, img_arr, dataset_path):
        self.bo_obj = BasicOperations()
        self.K_biggest = K_biggest
        self.eig_vectors_biggest = eig_vectors_biggest
        self.img_arr = img_arr
        self.dataset_path = dataset_path

    def calc_images_linear_comb(self):
        # calculating the images' coefficients
        self.coeff_matrix = np.dot(self.img_arr.T, self.eig_vectors_biggest).T

    def get_random_image(self):
        imgs_names = os.listdir(self.dataset_path)
        img_test = "*.txt"
        while (img_test.endswith(".txt")):
            rand_idx = np.random.randint(0, len(imgs_names))
            img_test = imgs_names[rand_idx]
        img = self.bo_obj.read_image(os.path.join(self.dataset_path, img_test))
        img = self.bo_obj.convert_img_colors("GS", img)
        print("input image - randomly:")
        self.bo_obj.show_img(img, True)
        img = img.flatten()
        img = img.reshape((img.shape[0], 1))
        img_params_name = "imgs_mean1.npy"
        mean_img = np.load(img_params_name)
        return (img - mean_img), mean_img

    def calc_linear_comb_distances(self, img):
        # calculating the coefficients for the given img, and finding the closest img (norm-2 on weights vectors)
        coeff_vec = np.dot(img.T, self.eig_vectors_biggest)
        coeff_vec = coeff_vec.reshape((self.K_biggest, 1))
        dist_matrix = np.zeros(self.coeff_matrix.shape[1])
        for i in range(self.coeff_matrix.shape[1]):
            tmp_col = self.coeff_matrix[:, i].reshape((self.K_biggest, 1))
            dist_matrix[i] = np.linalg.norm((coeff_vec - tmp_col), ord=2)
        return np.argmin(dist_matrix)

    def show_final_image(self, original_img, mean_img):
        # calculating the final image
        img_dim = int(np.sqrt(original_img.shape[0]))
        mean_img = mean_img.reshape((img_dim, img_dim)) 
        final_img = original_img.reshape((img_dim, img_dim)) + mean_img
        print("image found in the dataset:")
        self.bo_obj.show_img(final_img, True)


class Test:
    # class for tests
    def __init__(self):
        pass
    
    def test_get_data(self, dataset_path):
        ld_obj = LoadData(dataset_path)
        return ld_obj.build_faces_matrix()

    def test_show_eigenfaces(self, eig_vectors_biggest):
        bo_obj = BasicOperations()
        img_dim = int(np.sqrt(eig_vectors_biggest[:, 0].shape[0]))
        for i in range(10):
            tmp_img = eig_vectors_biggest[:, i]
            tmp_img = tmp_img.reshape((img_dim, img_dim)).real
            bo_obj.show_img(tmp_img, True)

    def test_pca(self, img_arr):
        pca_obj = PCA()
        K_biggest, eig_vectors_biggest = pca_obj.calc_pca_eigenfaces(img_arr)
        # self.test_show_eigenfaces(eig_vectors_biggest)
        return K_biggest, eig_vectors_biggest

    def test_recognition(self, K_biggest, eig_vectors_biggest, img_arr, dataset_path):
        rf_obj = RecognizeFaces(K_biggest, eig_vectors_biggest, img_arr, dataset_path)
        rf_obj.calc_images_linear_comb()
        img, mean_img = rf_obj.get_random_image()
        original_img = img_arr[:, rf_obj.calc_linear_comb_distances(img)]
        rf_obj.show_final_image(original_img, mean_img)


def main():
    dataset_path = "./simpson_dataset"
    tp = Test()
    img_arr = tp.test_get_data(dataset_path)
    if (type(img_arr) == bool): # when img_arr = False
        return False
    K_biggest, eig_vectors_biggest = tp.test_pca(img_arr)
    tp.test_recognition(K_biggest, eig_vectors_biggest, img_arr, dataset_path)
    

if __name__ == "__main__":
    main()