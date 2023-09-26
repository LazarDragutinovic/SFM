from sfm import  SFM

def main():
    sfm = SFM(".\\images", "camera_conf")
    sfm.Run()


if __name__ == "__main__":
    main()