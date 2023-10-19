class ImageData:
    R = None
    t = None
    refs = None
    def __init__(self, kp, des, mat):
        self.kp = kp
        self.des = des
        self.mat = mat