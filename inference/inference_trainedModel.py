import argparse
import glob
import multiprocessing as mp
from collections import deque
import cv2
import torch
import time
import tqdm
import torch.multiprocessing as mp

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

# from detectron2.engine.defaults import DefaultPredictor
# from detectron2.utils.video_visualizer import VideoVisualizer

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 inference_model")
    parser.add_argument(
        "--config-file",
        default="/raid01/qutub/projects/semantic_segmentation/mask-rcnn_pytorch/detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input", nargs="+", 
        default="/raid01/qutub/projects/semantic_segmentation/mask-rcnn_pytorch/inference/data/1280px-Korea-Gyeongju-Cityscape-Street_and_cars-02.jpg",
        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default="/raid01/qutub/projects/semantic_segmentation/mask-rcnn_pytorch/inference/data/output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--path-file",
        default="/raid01/qutub/projects/semantic_segmentation/mask-rcnn_pytorch/detectron2/output/model_final.pth",
        metavar="FILE",
        help="path to weights file",
    )    
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

class Visualizationinference_model(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.model = build_model(cfg)

    def load_weights(self, file_path):#="/raid01/qutub/projects/semantic_segmentation/mask-rcnn_pytorch/detectron2/output/model_final.pth")
        DetectionCheckpointer(self.model).load(file_path)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # vis_output = None
        # predictions = self.predictor(image)
        # # Convert image from OpenCV BGR format to Matplotlib RGB format.
        # image = image[:, :, ::-1]
        # visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        # if "panoptic_seg" in predictions:
        #     panoptic_seg, segments_info = predictions["panoptic_seg"]
        #     vis_output = visualizer.draw_panoptic_seg_predictions(
        #         panoptic_seg.to(self.cpu_device), segments_info
        #     )
        # else:
        #     if "sem_seg" in predictions:
        #         vis_output = visualizer.draw_sem_seg(
        #             predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
        #         )
        #     if "instances" in predictions:
        #         instances = predictions["instances"].to(self.cpu_device)
        #         vis_output = visualizer.draw_instance_predictions(predictions=instances)

        # return predictions, vis_output

        predictions = self.model(image)
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output        



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    #inference_model = VisualizationDemo(cfg)
    inference_model = Visualizationinference_model(cfg)
    inference_model.load_weights(args.path_file)
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        #for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation

        print("\nThis THIS \n")
        print(args.input)
        img = read_image(args.input, format="BGR")
        start_time = time.time()
        predictions, visualized_output = inference_model.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            # @ssq edit
            # if cv2.waitKey(0) == 27:
            #     break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(inference_model.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(inference_model.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()