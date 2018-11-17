import glob
from sift_testing import SIFT





if __name__ == "__main__":

    filepath = "data/"
    sigma = [5, 10, 20, 40, 100]

    for file in glob.glob(filepath + "*.jpg"):

        sift = SIFT(file)
        noisy_imgs = list(map(sift.add_noise, sigma))
        sift.getKpAndDescriptors(noisy_imgs[0])

