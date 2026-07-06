
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


# =================== Calibration (gaze_dir -> norm_xy and back) ===================
RIDGE_MODEL_PATH = "calibration_ridge_model.json"
RIDGE_MODEL_ABS_PATH = os.path.join(_PARENT_DIR, RIDGE_MODEL_PATH)


# =================== YOLO ===================
SEG_CONF = 0.15
SEG_IOU_THRESH = 0.50
SEG_MODEL_PATH = "yolov8n-seg.pt"
MAX_SEGMENTS_TO_RENDER = 80         # Cap drawn boxes only; scoring still uses all


# =================== CLIP + object DB ===================
# EVA-02-L/14 at 336px. Drop-in replacement for the older ViT-B/32; clearer
# score separation on the closed 3-object set. First load auto-downloads
# ~430 MB from HuggingFace.
CLIP_MODEL_NAME = "EVA02-L-14-336"
CLIP_PRETRAINED = "merged2b_s6b_b61k"

# Object DB lives in MacProgram/object_db/.
#   objects.json         metadata + per-object info text
#   images/<obj_id>/*    reference images for each object
#   embeddings.npz       cached CLIP embeddings, regenerated when images change
#                        OR when CLIP_MODEL_NAME/CLIP_PRETRAINED change.
OBJECT_DB_DIR = os.path.join(_PARENT_DIR, "object_db")
OBJECT_DB_JSON = os.path.join(OBJECT_DB_DIR, "objects.json")
OBJECT_DB_IMAGES_DIR = os.path.join(OBJECT_DB_DIR, "images")
OBJECT_DB_EMBEDDINGS_PATH = os.path.join(OBJECT_DB_DIR, "embeddings.npz")

# EVA-02 produces a tighter score distribution than ViT-B/32 -- matched pairs
# tend to land in 0.55-0.75 (with the embedding-space cosine being a bit lower
# than ViT-B/32 numerically). Keep masked_crop off per the reference paper.
CLIP_MATCH_MIN_SCORE = 0.55
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

# Structured after Figure 8 of GazePointAR (Lee et al., CHI '24). The paper's
# prompt has five bands: original query, gaze target, pointing target, other
# scene objects, and an answering rubric. We keep the same shape but adapt
# three things for our stack:
#   1. GazePointAR fed GPT-3 pre-extracted text (YOLO parent + OCR children).
#      We use GPT-5 vision, so instead of a phrase we ship the raw ADB
#      passthrough frame plus an EXPLICIT gaze pixel coordinate ({gaze_info})
#      that Unity computed from the user's eye tracker. The model uses that
#      coordinate to look at the right spot in the image itself.
#   2. Our pointing signal is implicit -- if the user's hand is in the frame
#      pointing at something, GPT-5 can see it directly. We call this out as
#      a fallback that outranks gaze only when clearly present.
#   3. Answer language follows the user's transcript language (Korean/English)
#      instead of always English.
VOICE_COMMAND_PROMPT = """\
You are a context-aware voice assistant for a user wearing an XR headset. The
image below is a snapshot of the user's real-world field of view (passthrough
camera + Unity overlays) captured the moment their speech was recognized.
Treat pronouns in the query (this, that, here, there, it, they, 이것, 저것,
여기, 저기 등) as pointers to something visible in that image.

===== 1. User query (verbatim) =====
The user asked: "{transcript}"

===== 2. Where the user was LOOKING (gaze target) =====
{gaze_info}

===== 3. Where the user is POINTING (secondary cue) =====
Inspect the frame for a visible hand, extended finger, or hand-held pointer.
If one is present and clearly aimed at an object, treat that object as the
referent instead of the gaze target -- explicit pointing outranks gaze. If
no pointing gesture is visible, ignore this section and rely on the gaze
target above.

===== 4. Other objects in view (peripheral context) =====
Everything else visible in the frame (background objects, text, signage,
overlays) may still matter for questions like "what else is here?" or
"which of these ...?", but weight it below the gaze / pointing target when
resolving a specific referent.

===== 5. Answer this question =====
"{transcript}"

===== 6. Output format (STRICT) =====
Respond with EXACTLY one JSON object, no prose before or after:

{
  "name":       "<short label for the resolved target or task>",
  "referent":   "<what you resolved any pronoun to, grounded in the gaze/pointing target, e.g. 'the blue soda can at the gaze pixel'; empty string if the query had no pronoun>",
  "answer":     "<ONE natural sentence that directly answers the user, followed by a short justification clause>",
  "confidence": "high" | "medium" | "low"
}

===== 7. Answering rules =====
- Answer in the SAME language the user spoke (Korean transcript -> Korean
  answer; English transcript -> English answer).
- Keep the tone natural and conversational, like answering a curious friend.
- The "answer" field must be a single sentence. Include a short "because ..."
  or "-- <reason>" clause so the user understands why.
- Ground your answer in what is actually visible AT OR NEAR the gaze pixel
  first. Do NOT invent objects that are not in the frame.
- If the object at the gaze target is unclear, admit it in "answer", set
  "confidence" to "low", and describe what IS at that pixel neighbourhood so
  the user can tell whether the tracker mis-aimed.
- Even with missing info or an ambiguous referent, DO NOT refuse. Give your
  best estimate or a range and set "confidence" to "low".
- If the query has no pronoun and does not refer to the visible scene at
  all (e.g. a general knowledge question), still answer using the same
  format, leave "referent" empty, and set "confidence" based on how sure you
  are of the general answer.
"""
