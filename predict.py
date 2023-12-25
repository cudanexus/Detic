import sys
import cv2
import tempfile
from pathlib import Path
import cog
from cog import BasePredictor, Path, Input,BaseModel

import time
import json

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

class Output(BaseModel):
    image: Path
    jsona: Path

class Predictor(BasePredictor):
    def setup(self):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.predictor = DefaultPredictor(cfg)
        self.BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
            'coco': 'datasets/metadata/coco_clip_a+cname.npy',
        }
        self.BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }

    # Input(
    #     "image",
    #     type=Path,
    #     help="input image",
    # )
    # Input(
    #     "vocabulary",
    #     type=str,
    #     default='lvis',
    #     options=['lvis', 'objects365', 'openimages', 'coco', 'custom'],
    #     help="Choose vocabulary",
    # )
    # Input(
    #     "custom_vocabulary",
    #     type=str,
    #     default=None,
    #     help="Type your own vocabularies, separated by coma ','",
    # )
    def predict(self, image: Path = Input(description="input image"),
                vocabulary:str =Input(description="Choose vocabulary"), 
                custom_vocabulary:str =Input(description="Choose vocabulary")) -> Output:
        image = cv2.imread(str(image))
        if not vocabulary == 'custom':
            metadata = MetadataCatalog.get(self.BUILDIN_METADATA_PATH[vocabulary])
            classifier = self.BUILDIN_CLASSIFIER[vocabulary]
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.predictor.model, classifier, num_classes)

        else:
            assert custom_vocabulary is not None and len(custom_vocabulary.split(',')) > 0, \
                "Please provide your own vocabularies when vocabulary is set to 'custom'."
            metadata = MetadataCatalog.get(str(time.time()))
            metadata.thing_classes = custom_vocabulary.split(',')
            classifier = get_clip_embeddings(metadata.thing_classes)
            num_classes = len(metadata.thing_classes)
            reset_cls_test(self.predictor.model, classifier, num_classes)
            # Reset visualization threshold
            output_score_threshold = 0.3
            for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
                self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold

        outputs = self.predictor(image)

        pred_classes = outputs["instances"].pred_classes.cpu().tolist()
        class_names = [metadata.thing_classes[x] for x in pred_classes]
        scores = outputs["instances"].scores.cpu().tolist()
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().tolist()

        # Create a dictionary to store the information
        output_dict = {
            "pred_classes": pred_classes,
            "class_names": class_names,
            "scores": scores,
            "pred_boxes": pred_boxes,
        }

        # Convert the dictionary to JSON
        json_output = json.dumps(output_dict, indent=2)

        # Specify the file path
        output_file_path = Path(tempfile.mkdtemp()) / "output.json"
        

        # Write the JSON data to the file
        with open(output_file_path, "w") as output_file:
            output_file.write(json_output)

        print(f"JSON data written to {output_file_path}")



        v = Visualizer(image[:, :, ::-1], metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), out.get_image()[:, :, ::-1])
        return Output(image=out_path,jsona=output_file_path)


def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb
