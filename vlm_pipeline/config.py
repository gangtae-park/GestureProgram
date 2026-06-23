
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
