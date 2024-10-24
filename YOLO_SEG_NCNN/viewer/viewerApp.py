import sys
import tkinter as tk
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageTk


class Dataset:
    def __init__(self, args, config):
        with open(config, 'r') as file:
            cfg = yaml.safe_load(file)
        self.datasets = sorted(glob('/'.join([cfg['path'], cfg[args.data], '*'])))
        self.targets = sorted(glob('/'.join([cfg['path'].replace('images', 'labels'), cfg[args.data], '*'])))

    def __getitem__(self, i):
        img_filename = self.datasets[i]

        img = cv2.imread(img_filename)
        #여기에 있는 이미지 로드해서 받아와
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(self.targets) != 0:
            label_filename = self.targets[i]
            with open(label_filename, 'r') as file:
                labels = file.readlines()
            if not labels:
                return img, None
            for label in labels:
                label = list(map(float, label.split()))
                cls = label[0] + 1
                contour = np.array(label[1:]).reshape(-1, 2)
                contour[:, 0] *= w
                contour[:, 1] *= h
                contour = np.array(contour, dtype=np.int32)
                cv2.fillPoly(mask, [contour], int(cls))
        else:
            mask = None
        return img, mask

    def __len__(self):
        return len(self.datasets)


class DemoApp:
    def __init__(self, args, config, model):
        self.datasets = Dataset(args, config)
        self.filenames = self.datasets.datasets
        self.current_index = 0

        self.model = model

        self.root = tk.Tk()
        self.root.title("SEG Demo")
        self.root.geometry("1300x900")

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        self.search_entry = tk.Entry(self.root, width=35)
        self.search_entry.pack(anchor='sw', padx=20)
        self.search_button = tk.Button(self.root, text="Search", command=self.handle_search)
        self.search_button.pack(anchor='sw', padx=20)

        self.infer = False
        self.infer_button = tk.Button(self.root, text="GT mode", command=self.convert_infer)
        self.infer_button.pack(anchor='se', padx=20)

        self.prev_button = tk.Button(self.root, text="Prev", command=self.infer_prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=20)

        self.next_button = tk.Button(self.root, text="Next", command=self.infer_next_image)
        self.next_button.pack(side=tk.RIGHT, padx=20)

        self.next_button = tk.Button(self.root, text="Convert", command=self.convert_image)
        self.next_button.pack(side=tk.RIGHT)
        self.toggle = True

        self.show_image(None)

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.root.mainloop()

    def on_exit(self):
        sys.exit()

    def handle_search(self):
        search_keyword = self.search_entry.get()
        if '.jpg' not in search_keyword:
            print('check your input')
            return
        print("filename:", search_keyword)
        index = None
        for i, filename in enumerate(self.filenames):
            if search_keyword == filename.split('/')[-1]:
                index = i
                break
        if index is not None:
            self.img_name_label.pack_forget()
            self.current_index = index
            if self.infer:
                mask = self.inference()
            else:
                mask = None
            self.show_image(mask)
        else:
            print('no file')

    def show_image(self, output):
        img, mask = self.datasets[self.current_index]
        # bg, da, ll
        colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255]])

        # draw infer
        if output is not None:
            cls, mask = output
            cls = cls.cpu().numpy().astype(bool)

            if mask is not None:
                mask_bg = torch.zeros(640, 640, device=mask.device)
                mask_da = mask[cls, :, :]
                mask_ll = mask[~cls, :, :]
                if mask_da.shape[0] == 0:
                    mask_da = mask_bg
                else:
                    mask_da, _ = torch.max(mask_da, dim=0)
                if mask_ll.shape[0] == 0:
                    mask_ll = mask_bg
                else:
                    mask_ll, _ = torch.max(mask_ll, dim=0)
                _, mask = torch.max(torch.stack([mask_bg, mask_ll, mask_da]), dim=0)
            else:
                mask = torch.zeros(640, 640)

            mask = mask.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # draw gt
        if mask is not None:
            mask_rgb = colormap[mask].astype(np.uint8)
            alpha = 0.6
            mask_rgb = cv2.addWeighted(img, alpha, mask_rgb, 1 - alpha, 0)

            img_copy = img.copy()
            img_copy[mask == 1] = mask_rgb[mask == 1]
            img_copy[mask == 2] = mask_rgb[mask == 2]
            mask = img_copy

        if self.toggle and mask is not None:
            img = mask

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img))
        photo = ImageTk.PhotoImage(img)

        self.img_name_label = tk.Label(self.root, text=self.datasets.datasets[self.current_index].split('/')[-1],
                                       font=("Arial", 12))

        self.img_name_label.pack(pady=5)

        self.image_label.config(image=photo)
        self.image_label.image = photo

    def infer_prev_image(self):
        self.img_name_label.pack_forget()
        self.current_index = (self.current_index - 1) % len(self.datasets)
        if self.infer and self.toggle:
            output = self.inference()
        else:
            output = None
        self.show_image(output)

    def infer_next_image(self):
        self.img_name_label.pack_forget()
        self.current_index = (self.current_index + 1) % len(self.datasets)
        if self.infer and self.toggle:
            output = self.inference()
        else:
            output = None
        self.show_image(output)

    def convert_infer(self):
        self.infer = not self.infer
        self.infer_button.config(text="Infer mode" if self.infer else "GT mode")

        self.img_name_label.pack_forget()
        self.current_index = self.current_index % len(self.datasets)
        if self.infer:
            output = self.inference()
        else:
            output = None
        self.show_image(output)

    def inference(self):
        img, _ = self.datasets[self.current_index]
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = torch.tensor(np.transpose(np.array(img, dtype=np.uint8) / 255, (2, 0, 1)), dtype=torch.float32)
        img = img.unsqueeze(0)
        with torch.no_grad():
            output = self.model(img)[0]
            #output = self.model(img)[0] <-여기 0은 batch size의 첫번째만 선택하겠다는 의미 (사진 1장만 선택!)
        return output.boxes.cls, output.masks.data if hasattr(output.masks, 'data') else None
        return output.boxes, output.masks.data if hasattr(output.masks, 'data') else None

    def convert_image(self):
        self.toggle = not self.toggle
        self.img_name_label.pack_forget()
        self.current_index = self.current_index % len(self.datasets)
        if self.infer and self.toggle:
            output = self.inference()
        else:
            output = None
        self.show_image(output)
