
import os

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PIPELINE_DIR)


# =================== UDP / Networking ===================
HOST = "0.0.0.0"
PORT = 5005                              # receive Unity packets

UNITY_RESULT_PORT = 5006                 # Unity listens for VLM results on this port
UNITY_HOST_OVERRIDE = None               # None = auto from inbound traffic
VLM_PACKET_PREFIX = "VLM_RESULT"         # Unity-side filter / matching prefix


# =================== Display / Canvas ===================
LIVE_WINDOW = "Live Stream"
TARGET_WINDOW = "Target Result"
CANVAS_W, CANVAS_H = 1100, 1000
STREAM_W, STREAM_H = 1100, 1000

POINT_RADIUS = 8
POINT_COLOR = (0, 0, 255)                # current gaze marker
TRAIL_COLOR = (0, 255, 255)              # in-progress gesture trail / gaze bbox
TARGET_COLOR = (0, 255, 0)               # YOLO target highlight
BG_COLOR = (30, 30, 30)


# =================== ADB Stream ===================
ADB_CMD = ["adb", "exec-out", "screenrecord", "--output-format=h264", "-"]


def build_ffmpeg_cmd(width: int, height: int):
    return [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "-",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}",
        "-",
    ]


# =================== Gaze / Gesture timing ===================
GAZE_BBOX_PADDING = 10
MIN_GAZE_POINTS_FOR_TARGET = 5
CAPTURE_DELAY_AFTER_END = 0.3

TARGET_SCORE_IOU_WEIGHT = 0.999
TARGET_MIN_OVERLAP = 0.03
TARGET_CROP_PAD_RATIO = 0.10             # 10% padding around target bbox for VLM crop

# Compare: number of objects to target, and a distinct overlay colour per slot.
COMPARE_TOP_N = 2
COMPARE_TARGET_COLORS = [(0, 255, 0), (0, 165, 255)]   # BGR: #1 green, #2 orange


# =================== Gaze Gaussian field (object targeting) ===================
# Object targeting no longer uses a rectangular gaze bbox. Instead the gaze
# trail becomes a soft field: each gaze point contributes a Gaussian blob and the
# blobs are accumulated into a dwell/fixation heatmap (then peak-normalised to
# [0,1]). Empty space between two separated objects stays near zero, and a brief
# saccade grazing across the gap barely registers, so targeting two distant
# objects (e.g. Compare) is not polluted by the gap a bounding box would include.
#
# A YOLO object is scored by the *soft IoU* (fuzzy Jaccard) between this field
# and the object's segmentation mask, not its box.
GAZE_GAUSSIAN_SIGMA_FRAC = 0.04          # blob std as a fraction of min(frame_h, frame_w)
GAZE_GAUSSIAN_MIN_SIGMA_PX = 12          # floor for the std on small frames
GAZE_GAUSSIAN_TRUNCATE = 3.0             # evaluate each blob within +/- this many std
GAZE_FIELD_HEATMAP_ALPHA = 0.6           # render: heatmap blend strength (0..1)
TARGET_MULTI_TOP_N = 2                   # how many objects Compare-style targeting returns


# =================== Calibration (gaze_dir -> norm_xy) ===================
RIDGE_MODEL_PATH = "calibration_ridge_model.json"


# =================== YOLO ===================
SEG_CONF = 0.15
SEG_IOU_THRESH = 0.50
SEG_MODEL_PATH = "yolov8n-seg.pt"
MAX_SEGMENTS_TO_RENDER = 80         # Cap drawn boxes only; scoring still uses all


# =================== CLIP + object DB ===================
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Object DB lives in MacProgram/object_db/.
#   objects.json         metadata + per-object info text
#   images/<obj_id>/*    reference images for each object
#   embeddings.npz       cached CLIP embeddings, regenerated when images change
OBJECT_DB_DIR = os.path.join(_PARENT_DIR, "object_db")
OBJECT_DB_JSON = os.path.join(OBJECT_DB_DIR, "objects.json")
OBJECT_DB_IMAGES_DIR = os.path.join(OBJECT_DB_DIR, "images")
OBJECT_DB_EMBEDDINGS_PATH = os.path.join(OBJECT_DB_DIR, "embeddings.npz")

CLIP_MATCH_MIN_SCORE = 0.7
CLIP_USE_MASKED_CROP = False


# =================== VLM (OpenAI GPT) ===================
OPENAI_MODEL = "gpt-5"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
VLM_OUTPUT_DIR = os.path.join(_PARENT_DIR, "vlm_outputs")
VLM_REQUEST_TIMEOUT_SEC = 30
VLM_MAX_OUTPUT_TOKENS = 2000
VLM_REASONING_EFFORT = "low"


# =================== OCR / Translate ===================
OCR_ROI_PAD_PX = 240
OCR_ROI_MIN_SIDE = 480
OCR_PARAGRAPH = True
OCR_MAX_SIDE = 960
OCR_MIN_CONF = 0.30


# =================== PROMPTS ===================

ASK_REFERENCE_PROMPT = """\
You are answering a user's open-ended question about a real-world object they pointed at in XR.
The image is a tight crop around the object. The object has already been identified for you
in a "DB info" block below -- treat that as authoritative ground truth and only fall back to
the image for things the DB does not cover (colour, condition, position, etc.).

Respond ONLY with a single JSON object using this schema:

{
  "name": "<short object name -- prefer the DB name>",
  "answer": "<answer to the question about the object>"
}

Style:
- Keep the tone natural and conversational, like answering a curious friend.
- The 'answer' field must be a complete sentence.
- If the question can't be answered from the DB or the image, say so plainly
  instead of guessing.
"""
