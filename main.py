import os.path
import time

from model_generator import ModelGenerator
import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import showinfo

def main():
    root = tk.Tk()
    root.title("3d model generator")
    root.geometry("1200x350")
    root.resizable(False, False)

    config = dict()

    def add_image_folder_name():
        config['image_folder_name'] = filedialog.askdirectory()

    def add_conf_file_name():
        config['conf_file_name'] = filedialog.askopenfilename()

    destination_entry = tk.Entry(root, width=50, borderwidth=5)

    def add_destination_directory():
        config['destination_name'] = filedialog.askdirectory()
        if config['destination_name'] != None:
            destination_entry.grid(row=0, column=4, padx=10, pady=5)

    def run_reconstruction():
        image_folder_name = config['image_folder_name']
        conf_file_name = config['conf_file_name']
        destination_name = config['destination_name']

        start = time.time()
        model_gen = ModelGenerator(
            image_folder_name, conf_file_name, os.path.join(destination_name, destination_entry.get() + ".ply")
        )
        status = model_gen.Run()
        end = time.time()
        status = 'total time: ' + str(end - start) + 's\n' + status
        tk.Label(root, text=status).grid(row=1, column=0, columnspan=2)
        showinfo(title="Reconstruction status", message="Reconstruction finished successfuly.")

    tk.Button(root, text="Select image directory", command=add_image_folder_name).grid(row=0, column=0, columnspan=2, padx=10, pady=5)
    tk.Button(root, text="Select conf file", command=add_conf_file_name).grid(row=0, column=2, padx=10, pady=5)
    tk.Button(root, text="Select destination directory", command=add_destination_directory).grid(row=0, column=3, padx=10, pady=5)
    tk.Button(root, text="Run reconstruction", command=run_reconstruction).grid(row=0, column=5, padx=10, pady=5)
    root.mainloop()

if __name__ == "__main__":
    main()