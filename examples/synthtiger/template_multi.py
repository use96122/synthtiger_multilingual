"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
import json

from synthtiger import components, layers, templates, utils

shard = 4
BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "hard_light",
    "soft_light",
    "dodge",
    "divide",
    "addition",
    "difference",
    "darken_only",
    "lighten_only",
]


class SynthTiger(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.coord_output = config.get("coord_output", True)
        self.mask_output = config.get("mask_output", True)
        self.glyph_coord_output = config.get("glyph_coord_output", True)
        self.glyph_mask_output = config.get("glyph_mask_output", True)
        self.vertical = config.get("vertical", False)
        self.quality = config.get("quality", [95, 95])
        self.visibility_check = config.get("visibility_check", False)
        self.midground = config.get("midground", 0)
        self.midground_offset = components.Translate(
            **config.get("midground_offset", {})
        )
        self.foreground_mask_pad = config.get("foreground_mask_pad", 0)
        self.kor_corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("korean_corpus", {}),
        )
        self.eng_corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("english_corpus", {}),
        )
        self.chi_corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("chinese_corpus", {}),
        )
        self.arab_corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("arabic_corpus", {}),
        )
        
        self.thai_corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("thai_corpus", {}),
        )
        
        self.russian_corpus = components.Selector(
            [
                components.LengthAugmentableCorpus(),
                components.CharAugmentableCorpus(),
            ],
            **config.get("russian_corpus", {}),
        )
        
        self.kor_font = components.BaseFont(**config.get("korean_font", {}))
        self.eng_font = components.BaseFont(**config.get("english_font", {}))
        self.chi_font = components.BaseFont(**config.get("chinese_font", {}))
        self.arab_font = components.BaseFont(**config.get("arabic_font", {}))
        self.thai_font = components.BaseFont(**config.get("thai_font", {}))
        self.russian_font = components.BaseFont(**config.get("russian_font", {}))
        
        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {})
        )
        self.colormap2 = components.GrayMap(**config.get("colormap2", {}))
        self.colormap3 = components.GrayMap(**config.get("colormap3", {}))
        self.color = components.Gray(**config.get("color", {}))
        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {}),
        )
        self.layout = components.Selector(
            [components.FlowLayout(), components.CurveLayout()],
            **config.get("layout", {}),
        )
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )
        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )
        self.fit = components.Fit()
        self.pad = components.Switch(components.Pad(), **config.get("pad", {}))
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )

    def generate(self):
        print("generate..")
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        midground = np.random.rand() < self.midground
        fg_color, fg_style, mg_color, mg_style, bg_color = self._generate_color()
        
        eng_result, kor_result, chi_result, arab_result, thai_result, russian_result = self._generate_text(
            fg_color, fg_style
        )

        eng_fg_image, eng_label, eng_bboxes, eng_glyph_fg_image, eng_glyph_bboxes = eng_result
        kor_fg_image, kor_label, kor_bboxes, kor_glyph_fg_image, kor_glyph_bboxes = kor_result
        chi_fg_image, chi_label, chi_bboxes, chi_glyph_fg_image, chi_glyph_bboxes = chi_result
        arab_fg_image, arab_label, arab_bboxes, arab_glyph_fg_image, arab_glyph_bboxes = arab_result
        thai_fg_image, thai_label, thai_bboxes, thai_glyph_fg_image, thai_glyph_bboxes = thai_result
        russian_fg_image, russian_label, russian_bboxes, russian_glyph_fg_image, russian_glyph_bboxes = russian_result

        
        eng_fg_image_shape = eng_fg_image.shape[:2][::-1]
        kor_fg_image_shape = kor_fg_image.shape[:2][::-1]
        chi_fg_image_shape = chi_fg_image.shape[:2][::-1]
        arab_fg_image_shape = arab_fg_image.shape[:2][::-1]
        thai_fg_image_shape = thai_fg_image.shape[:2][::-1]
        russian_fg_image_shape = russian_fg_image.shape[:2][::-1]

        eng_bg_image, kor_bg_image, chi_bg_image, arab_bg_image, thai_bg_image, russian_bg_image = self._generate_background(eng_fg_image_shape, kor_fg_image_shape, chi_fg_image_shape, arab_fg_image_shape, thai_fg_image_shape, russian_fg_image_shape, bg_color)
        
        try :
            kor_image, eng_image, chi_image, arab_image, thai_image, russian_image = _blend_images(kor_fg_image, kor_bg_image, eng_fg_image, eng_bg_image, chi_fg_image, chi_bg_image, arab_fg_image, arab_bg_image, thai_fg_image,thai_bg_image, russian_fg_image,russian_bg_image,self.visibility_check)
        except :
            return 0
        
        kor_data = {
            "language" : "korean",
            "image": kor_image,
            "label": kor_label,
            "quality": quality,
            "mask": kor_fg_image[..., 3],
            "bboxes": kor_bboxes,
            "glyph_mask": kor_glyph_fg_image[..., 3],
            "glyph_bboxes": kor_glyph_bboxes,
        }
        eng_data = {
            "language" : "english",
            "image": eng_image,
            "label": eng_label,
            "quality": quality,
            "mask": eng_fg_image[..., 3],
            "bboxes": eng_bboxes,
            "glyph_mask": eng_glyph_fg_image[..., 3],
            "glyph_bboxes": eng_glyph_bboxes,
        }
        chi_data = {
            "language" : "chinese",
            "image": chi_image,
            "label": chi_label,
            "quality": quality,
            "mask": chi_fg_image[..., 3],
            "bboxes": chi_bboxes,
            "glyph_mask": chi_glyph_fg_image[..., 3],
            "glyph_bboxes": chi_glyph_bboxes,
        }
        arab_data = {
            "language" : "arabic",
            "image": arab_image,
            "label": arab_label,
            "quality": quality,
            "mask": arab_fg_image[..., 3],
            "bboxes": arab_bboxes,
            "glyph_mask": arab_glyph_fg_image[..., 3],
            "glyph_bboxes": arab_glyph_bboxes,
        }
        thai_data = {
            "language" : "thailand",
            "image": thai_image,
            "label": thai_label,
            "quality": quality,
            "mask": thai_fg_image[..., 3],
            "bboxes": thai_bboxes,
            "glyph_mask": thai_glyph_fg_image[..., 3],
            "glyph_bboxes": thai_glyph_bboxes,
        }
        russian_data = {
            "language" : "russian",
            "image": russian_image,
            "label": russian_label,
            "quality": quality,
            "mask": russian_fg_image[..., 3],
            "bboxes": russian_bboxes,
            "glyph_mask": russian_glyph_fg_image[..., 3],
            "glyph_bboxes": russian_glyph_bboxes,
        }


        return kor_data, eng_data, chi_data, arab_data, thai_data, russian_data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

        gt_path = os.path.join(root, "gt.txt")
        coords_path = os.path.join(root, "coords.txt")
        glyph_coords_path = os.path.join(root, "glyph_coords.json")

        self.gt_file = open(gt_path, "w", encoding="utf-8")
        if self.coord_output:
            self.coords_file = open(coords_path, "w", encoding="utf-8")
        if self.glyph_coord_output:
            # JSON 데이터를 저장할 리스트 초기화
            self.glyph_coords = []

    def save(self, root, data, idx):
        if data != 0:
            for i in range(len(data)):
                lan = data[i]
                language = lan["language"]
                image = lan["image"]
                label = lan["label"]
                quality = lan["quality"]
                mask = lan["mask"]
                bboxes = lan["bboxes"]
                glyph_mask = lan["glyph_mask"]
                glyph_bboxes = lan["glyph_bboxes"]

                image = Image.fromarray(image[..., :3].astype(np.uint8))
                mask = Image.fromarray(mask.astype(np.uint8))
                glyph_mask = Image.fromarray(glyph_mask.astype(np.uint8))
                    
                vis_image_pil = image.copy()
                draw = ImageDraw.Draw(vis_image_pil)

                # 글자 bboxes를 통해 단어 bbox 추론
                x_min = min([float(bbox[0]) for bbox in glyph_bboxes])
                y_min = min([float(bbox[1]) for bbox in glyph_bboxes])
                x_max = max([float(bbox[0]) + float(bbox[2]) for bbox in glyph_bboxes])
                y_max = max([float(bbox[1]) + float(bbox[3]) for bbox in glyph_bboxes])

                word_bbox = {
                    "label": label,  
                    "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                }

                # 단어 bbox 시각화
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

                # 각 글자 bbox 시각화
                for j, glyph_bbox in enumerate(glyph_bboxes):
                    glyph_x_min = float(glyph_bbox[0])
                    glyph_y_min = float(glyph_bbox[1])
                    glyph_x_max = glyph_x_min + float(glyph_bbox[2])
                    glyph_y_max = glyph_y_min + float(glyph_bbox[3])

                    draw.rectangle([glyph_x_min, glyph_y_min, glyph_x_max, glyph_y_max], outline="blue", width=2)
                base_idx = shard * 10000 + idx
                
                # 단어의 bbox와 각 character의 bbox를 포함한 JSON 객체 생성
                word_data = {
                    "language" : language,
                    "idx": base_idx,  # 추가된 idx 값
                    "word_bbox": word_bbox,
                    "characters": [
                        {
                            "character": label[j],
                            "bbox": [float(glyph_bbox[0]), float(glyph_bbox[1]), float(glyph_bbox[2]), float(glyph_bbox[3])]
                        } for j, glyph_bbox in enumerate(glyph_bboxes)
                    ]
                }

                if self.glyph_coord_output:
                    # JSON 데이터를 리스트에 추가
                    self.glyph_coords.append(word_data)

                # shard = str(idx // 10000)
                # image_key = os.path.join("images", shard, f"{idx}.jpg")
                # mask_key = os.path.join("masks", shard, f"{idx}.png")
                # glyph_mask_key = os.path.join("glyph_masks", shard, f"{idx}.png")
                
                
                image_key = os.path.join("images", str(shard), f"{base_idx}.jpg")
                mask_key = os.path.join("masks", str(shard), f"{base_idx}.png")
                glyph_mask_key = os.path.join("glyph_masks", str(shard), f"{base_idx}.png")
                
                
                # vis_image_key = os.path.join("visualizations", shard, f"{idx}_vis.jpg")

                if i == 0:
                    language = "korean"
                elif i == 1:
                    language = "english"
                elif i == 2:
                    language = "chinese"
                elif i == 3:
                    language = "arabic"
                elif i == 4:
                    language = "thai"
                else:
                    language = "russian"

                image_path = os.path.join(root, language, image_key)
                mask_path = os.path.join(root, language, mask_key)
                glyph_mask_path = os.path.join(root, language, glyph_mask_key)
                # vis_image_path = os.path.join(root, language, vis_image_key)

                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path, quality=quality)
                if self.mask_output:
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    mask.save(mask_path)
                if self.glyph_mask_output:
                    os.makedirs(os.path.dirname(glyph_mask_path), exist_ok=True)
                    glyph_mask.save(glyph_mask_path)
                        
                # os.makedirs(os.path.dirname(vis_image_path), exist_ok=True)
                # vis_image_pil.save(vis_image_path, quality=quality)

                self.gt_file.write(f"{image_key}\t{label}\n")
                if self.coord_output:
                    self.coords_file.write(f"{image_key}\t{word_bbox['bbox'][0]},{word_bbox['bbox'][1]},{word_bbox['bbox'][2]},{word_bbox['bbox'][3]}\n")
                
                print(label)
                print(image_path)
                

    def end_save(self, root):
        self.gt_file.close()
        if self.coord_output:
            self.coords_file.close()
        if self.glyph_coord_output:
            with open(os.path.join(root, f"glyph_coords_{shard}.json"), "w", encoding="utf-8") as json_file:
                json.dump(self.glyph_coords, json_file, ensure_ascii=False, indent=4)

    def _generate_color(self):
        mg_color = self.color.sample()
        fg_style = self.style.sample()
        mg_style = self.style.sample()

        if fg_style["state"]:
            fg_color, bg_color, style_color = self.colormap3.sample()
            fg_style["meta"]["meta"]["rgb"] = style_color["rgb"]
        else:
            fg_color, bg_color = self.colormap2.sample()

        return fg_color, fg_style, mg_color, mg_style, bg_color

    def _generate_text(self, color, style):
        eng_label = self.eng_corpus.data(self.eng_corpus.sample())
        eng_chars = utils.split_text(eng_label, reorder=True)
        
        kor_label = self.kor_corpus.data(self.kor_corpus.sample())
        kor_chars = utils.split_text(kor_label, reorder=True)
        
        chi_label = self.chi_corpus.data(self.chi_corpus.sample())
        chi_chars = utils.split_text(chi_label, reorder=True)
        
        arab_label = self.arab_corpus.data(self.arab_corpus.sample())
        arab_chars = utils.split_text(arab_label, reorder=True)
        
        thai_label = self.thai_corpus.data(self.thai_corpus.sample())
        thai_chars = utils.split_text(thai_label, reorder=True)
        
        russian_label = self.russian_corpus.data(self.russian_corpus.sample())
        russian_chars = utils.split_text(russian_label, reorder=True)


        eng_text = "".join(eng_chars)
        eng_font = self.eng_font.sample({"text": eng_text, "vertical": self.vertical})
        
        kor_text = "".join(kor_chars)
        kor_font = self.kor_font.sample({"text": kor_text, "vertical": self.vertical})
        
        chi_text = "".join(chi_chars)
        chi_font = self.chi_font.sample({"text": chi_text, "vertical": self.vertical})
        
        arab_text = "".join(arab_chars)
        arab_font = self.arab_font.sample({"text": arab_text, "vertical": self.vertical})
        
        thai_text = "".join(thai_chars)
        thai_font = self.thai_font.sample({"text": thai_text, "vertical": self.vertical})
        
        russian_text = "".join(russian_chars)
        russian_font = self.russian_font.sample({"text": russian_text, "vertical": self.vertical})

        

        eng_char_layers = [layers.TextLayer(char, **eng_font) for char in eng_chars]
        kor_char_layers = [layers.TextLayer(char, **kor_font) for char in kor_chars]
        chi_char_layers = [layers.TextLayer(char, **chi_font) for char in chi_chars]
        arab_char_layers = [layers.TextLayer(char, **arab_font) for char in arab_chars]
        thai_char_layers = [layers.TextLayer(char, **thai_font) for char in thai_chars]
        russian_char_layers = [layers.TextLayer(char, **russian_font) for char in russian_chars]

        
        #self.shape.apply(char_layers for char_layers in [eng_char_layers,kor_char_layers,chi_char_layers,arab_char_layers])
        #self.layout.apply([eng_char_layers,kor_char_layers,chi_char_layers,arab_char_layers], {"meta": {"vertical": self.vertical}})
        #list(map(lambda layers: self.layout.apply(layers, {"meta": {"vertical": self.vertical}}), [eng_char_layers, kor_char_layers, chi_char_layers, arab_char_layers]))

        
        # layout
        self.layout.apply(eng_char_layers, {"meta": {"vertical": self.vertical}})
        self.layout.apply(kor_char_layers, {"meta": {"vertical": self.vertical}})
        self.layout.apply(chi_char_layers, {"meta": {"vertical": self.vertical}})
        self.layout.apply(arab_char_layers, {"meta": {"vertical": self.vertical}})
        self.layout.apply(thai_char_layers, {"meta": {"vertical": self.vertical}})
        self.layout.apply(russian_char_layers, {"meta": {"vertical": self.vertical}})
        
        
        eng_char_glyph_layers = [char_layer.copy() for char_layer in eng_char_layers]
        kor_char_glyph_layers = [char_layer.copy() for char_layer in kor_char_layers]
        chi_char_glyph_layers = [char_layer.copy() for char_layer in chi_char_layers]
        arab_char_glyph_layers = [char_layer.copy() for char_layer in arab_char_layers]
        thai_char_glyph_layers = [char_layer.copy() for char_layer in thai_char_layers]
        russian_char_glyph_layers = [char_layer.copy() for char_layer in russian_char_layers]
        

        eng_text_layer = layers.Group(eng_char_layers).merge()
        eng_text_glyph_layer = eng_text_layer.copy()
        
        kor_text_layer = layers.Group(kor_char_layers).merge()
        kor_text_glyph_layer = kor_text_layer.copy()
        
        chi_text_layer = layers.Group(chi_char_layers).merge()
        chi_text_glyph_layer = chi_text_layer.copy()
        
        arab_text_layer = layers.Group(arab_char_layers).merge()
        arab_text_glyph_layer = arab_text_layer.copy()
        
        thai_text_layer = layers.Group(thai_char_layers).merge()
        thai_text_glyph_layer = thai_text_layer.copy()
        
        russian_text_layer = layers.Group(russian_char_layers).merge()
        russian_text_glyph_layer = russian_text_layer.copy()


        transform = self.transform.sample()

        texture = self.texture.sample()
        fit = self.fit.sample()
        pad = self.pad.sample()
        
        #  layer 적용
        self.color.apply(
            [eng_text_layer, eng_text_glyph_layer, 
            kor_text_layer, kor_text_glyph_layer, 
            chi_text_layer, chi_text_glyph_layer, 
            arab_text_layer, arab_text_glyph_layer, 
            thai_text_layer, thai_text_glyph_layer, 
            russian_text_layer, russian_text_glyph_layer],
            color
        )
        self.texture.apply(
            [eng_text_layer, eng_text_glyph_layer, 
            kor_text_layer, kor_text_glyph_layer, 
            chi_text_layer, chi_text_glyph_layer, 
            arab_text_layer, arab_text_glyph_layer, 
            thai_text_layer, thai_text_glyph_layer, 
            russian_text_layer, russian_text_glyph_layer],
            texture
        )
        self.style.apply(
            [eng_text_layer, *eng_char_layers, 
            kor_text_layer, *kor_char_layers, 
            chi_text_layer, *chi_char_layers, 
            arab_text_layer, *arab_char_layers, 
            thai_text_layer, *thai_char_layers, 
            russian_text_layer, *russian_char_layers],
            style
        )
        self.transform.apply(
            [eng_text_layer, eng_text_glyph_layer, *eng_char_layers, *eng_char_glyph_layers, 
            kor_text_layer, kor_text_glyph_layer, *kor_char_layers, *kor_char_glyph_layers,
            chi_text_layer, chi_text_glyph_layer, *chi_char_layers, *chi_char_glyph_layers, 
            arab_text_layer, arab_text_glyph_layer, *arab_char_layers, *arab_char_glyph_layers,
            thai_text_layer, thai_text_glyph_layer, *thai_char_layers, *thai_char_glyph_layers,
            russian_text_layer, russian_text_glyph_layer, *russian_char_layers, *russian_char_glyph_layers],
            transform
        )
        self.fit.apply(
            [eng_text_layer, eng_text_glyph_layer, *eng_char_layers, *eng_char_glyph_layers, 
            kor_text_layer, kor_text_glyph_layer, *kor_char_layers, *kor_char_glyph_layers, 
            chi_text_layer, chi_text_glyph_layer, *chi_char_layers, *chi_char_glyph_layers, 
            arab_text_layer, arab_text_glyph_layer, *arab_char_layers, *arab_char_glyph_layers,
            thai_text_layer, thai_text_glyph_layer, *thai_char_layers, *thai_char_glyph_layers,
            russian_text_layer, russian_text_glyph_layer, *russian_char_layers, *russian_char_glyph_layers],
            fit
        )
        self.pad.apply(
            [eng_text_layer, kor_text_layer, chi_text_layer, arab_text_layer, thai_text_layer, russian_text_layer],
            pad
        )

        for char_layer in eng_char_layers:
            char_layer.topleft -= eng_text_layer.topleft
        for char_glyph_layer in eng_char_glyph_layers:
            char_glyph_layer.topleft -= eng_text_layer.topleft
            
        for char_layer in kor_char_layers:
            char_layer.topleft -= kor_text_layer.topleft
        for char_glyph_layer in kor_char_glyph_layers:
            char_glyph_layer.topleft -= kor_text_layer.topleft
            
        for char_layer in chi_char_layers:
            char_layer.topleft -= chi_text_layer.topleft
        for char_glyph_layer in chi_char_glyph_layers:
            char_glyph_layer.topleft -= chi_text_layer.topleft
            
        for char_layer in arab_char_layers:
            char_layer.topleft -= arab_text_layer.topleft
        for char_glyph_layer in arab_char_glyph_layers:
            char_glyph_layer.topleft -= arab_text_layer.topleft
            
        for char_layer in thai_char_layers:
            char_layer.topleft -= thai_text_layer.topleft
        for char_glyph_layer in thai_char_glyph_layers:
            char_glyph_layer.topleft -= thai_text_layer.topleft
            
        for char_layer in russian_char_layers:
            char_layer.topleft -= russian_text_layer.topleft
        for char_glyph_layer in russian_char_glyph_layers:
            char_glyph_layer.topleft -= russian_text_layer.topleft


        eng_out = eng_text_layer.output()
        eng_bboxes = [char_layer.bbox for char_layer in eng_char_layers]

        eng_glyph_out = eng_text_glyph_layer.output(bbox=eng_text_layer.bbox)
        eng_glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in eng_char_glyph_layers]
        

        kor_out = kor_text_layer.output()
        kor_bboxes = [char_layer.bbox for char_layer in kor_char_layers]

        kor_glyph_out = kor_text_glyph_layer.output(bbox=kor_text_layer.bbox)
        kor_glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in kor_char_glyph_layers]
        
        chi_out = chi_text_layer.output()
        chi_bboxes = [char_layer.bbox for char_layer in chi_char_layers]

        chi_glyph_out = chi_text_glyph_layer.output(bbox=chi_text_layer.bbox)
        chi_glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in chi_char_glyph_layers]
        
        arab_out = arab_text_layer.output()
        arab_bboxes = [char_layer.bbox for char_layer in arab_char_layers]

        arab_glyph_out = arab_text_glyph_layer.output(bbox=arab_text_layer.bbox)
        arab_glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in arab_char_glyph_layers]
        
        thai_out = thai_text_layer.output()
        thai_bboxes = [char_layer.bbox for char_layer in thai_char_layers]

        thai_glyph_out = thai_text_glyph_layer.output(bbox=thai_text_layer.bbox)
        thai_glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in thai_char_glyph_layers]
        
        russian_out = russian_text_layer.output()
        russian_bboxes = [char_layer.bbox for char_layer in russian_char_layers]

        russian_glyph_out = russian_text_glyph_layer.output(bbox=russian_text_layer.bbox)
        russian_glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in russian_char_glyph_layers]


        return [eng_out, eng_label, eng_bboxes, eng_glyph_out, eng_glyph_bboxes],[kor_out, kor_label, kor_bboxes, kor_glyph_out, kor_glyph_bboxes], [chi_out, chi_label, chi_bboxes, chi_glyph_out, chi_glyph_bboxes], [arab_out, arab_label, arab_bboxes, arab_glyph_out, arab_glyph_bboxes], [thai_out, thai_label, thai_bboxes, thai_glyph_out, thai_glyph_bboxes], [russian_out, russian_label, russian_bboxes, russian_glyph_out, russian_glyph_bboxes]

    def _generate_background(self, eng_size, kor_size, chi_size, arab_size, thai_size,russian_size, color):
        eng_layer = layers.RectLayer(eng_size)
        kor_layer = layers.RectLayer(kor_size)
        chi_layer = layers.RectLayer(chi_size)
        arab_layer = layers.RectLayer(arab_size)
        thai_layer = layers.RectLayer(thai_size)
        russian_layer = layers.RectLayer(russian_size)
        
        color = self.color.sample()
        texture = self.texture.sample()
        
        self.color.apply([eng_layer,kor_layer, chi_layer, arab_layer, thai_layer, russian_layer], color)
        self.texture.apply([eng_layer,kor_layer, chi_layer, arab_layer, thai_layer, russian_layer], texture)
        
        eng_out = eng_layer.output()
        kor_out = kor_layer.output()
        chi_out = chi_layer.output()
        arab_out = arab_layer.output()
        thai_out = thai_layer.output()
        russian_out = russian_layer.output()
        
        return eng_out, kor_out, chi_out, arab_out, thai_out, russian_out

    def _erase_image(self, image, mask):
        mask = _create_poly_mask(mask, self.foreground_mask_pad)
        mask_layer = layers.Layer(mask)
        image_layer = layers.Layer(image)
        image_layer.bbox = mask_layer.bbox
        self.midground_offset.apply([image_layer])
        out = image_layer.erase(mask_layer).output(bbox=mask_layer.bbox)
        return out

    def _postprocess_images(self, images):
        image_layers = [layers.Layer(image) for image in images]
        self.postprocess.apply(image_layers)
        outs = [image_layer.output() for image_layer in image_layers]
        return outs

def _find_first_successful_blend_mode(src, dst, blend_modes, visibility_check):
    for blend_mode in blend_modes:
        out = utils.blend_image(src, dst, mode=blend_mode)
        if not visibility_check or _check_visibility(out, src[..., 3]):
            return blend_mode
    raise RuntimeError("Text is not visible in any blend mode")

def _blend_images(kor_src, kor_dst, eng_src, eng_dst, chi_src, chi_dst, arab_src, arab_dst, thai_src, thai_dst, russian_src, russian_dst, visibility_check=False):
    blend_modes = np.random.permutation(BLEND_MODES)

    successful_blend_mode = _find_first_successful_blend_mode(kor_src, kor_dst, blend_modes, visibility_check)

    kor_out = utils.blend_image(kor_src, kor_dst, mode=successful_blend_mode)
    eng_out = utils.blend_image(eng_src, eng_dst, mode=successful_blend_mode)
    chi_out = utils.blend_image(chi_src, chi_dst, mode=successful_blend_mode)
    arab_out = utils.blend_image(arab_src, arab_dst, mode=successful_blend_mode)
    thai_out = utils.blend_image(thai_src, thai_dst, mode=successful_blend_mode)
    russian_out = utils.blend_image(russian_src, russian_dst, mode=successful_blend_mode)

    return kor_out, eng_out, chi_out, arab_out, thai_out, russian_out

def _check_visibility(image, mask):
    gray = utils.to_gray(image[..., :3]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1


def _create_poly_mask(image, pad=0):
    height, width = image.shape[:2]
    alpha = image[..., 3].astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)

    cts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts = sorted(cts, key=lambda ct: sum(cv2.boundingRect(ct)[:2]))

    if len(cts) == 1:
        hull = cv2.convexHull(cts[0])
        cv2.fillConvexPoly(mask, hull, 255)

    for idx in range(len(cts) - 1):
        pts = np.concatenate((cts[idx], cts[idx + 1]), axis=0)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    mask = utils.dilate_image(mask, pad)
    out = utils.create_image((width, height))
    out[..., 3] = mask
    return out
